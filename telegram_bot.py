from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sys
from dataclasses import dataclass
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from predicciones.src.models.parlay_builder import build_all_same_game_parlays, check_calibration  # noqa: E402
from predicciones.src.pipeline.predict import predict_match_pipeline  # noqa: E402
from predicciones.src.telegram_integration import (  # noqa: E402
    available_competitions,
    build_match_items,
    build_stage_items,
    format_parlay_text,
    format_match_analysis_text,
    format_player_detail,
    format_prediction_text,
    format_timeline_text,
    get_competition_teams_and_matches,
    get_player_rows_for_team,
    load_calibration_manager,
    load_worldcup_matches,
    _stage_title,
    split_telegram_message,
)
from predicciones.src.utils.team_normalization import normalize_team_name  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

ROOT, PICK = range(2)
PAGE_SIZE = 6
REPORT_LIMIT = 3900
LOCAL_DERIVED_DIR = project_root / "data" / "derived"
LOCAL_TEAM_STATS_PATH = LOCAL_DERIVED_DIR / "team_match_stats.jsonl"
WORLD_CUP_BRACKET_URL = "https://www.espn.com/soccer/bracket/_/season/2026/league/fifa.world?xhr=1"


def main_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🔮 Prediccion", callback_data="menu|prediction")],
            [InlineKeyboardButton("🧩 Generar Parlay", callback_data="menu|parlay")],
            [InlineKeyboardButton("⏱️ Time lines de partidos", callback_data="menu|timelines")],
            [InlineKeyboardButton("📊 Analisis de Partidos", callback_data="menu|analysis")],
            [InlineKeyboardButton("👤 Datos de jugadores", callback_data="menu|players")],
        ]
    )


def navigation_markup() -> List[List[InlineKeyboardButton]]:
    return [
        [
            InlineKeyboardButton("↩️ Volver al menú", callback_data="nav|back"),
            InlineKeyboardButton("🏠 Menú principal", callback_data="nav|menu"),
            InlineKeyboardButton("Cancelar", callback_data="nav|cancel"),
        ]
    ]


def build_picker_markup(
    picker_id: str,
    items: List[Dict[str, Any]],
    page: int = 0,
    show_back: bool = True,
) -> InlineKeyboardMarkup:
    start = page * PAGE_SIZE
    page_items = items[start:start + PAGE_SIZE]
    rows: List[List[InlineKeyboardButton]] = []

    for index in range(0, len(page_items), 2):
        row = []
        for offset, item in enumerate(page_items[index:index + 2]):
            absolute_idx = start + index + offset
            row.append(
                InlineKeyboardButton(
                    item["label"][:48],
                    callback_data=f"pick|{picker_id}|{absolute_idx}",
                )
            )
        rows.append(row)

    total_pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
    nav_row: List[InlineKeyboardButton] = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("Anterior", callback_data=f"page|{picker_id}|{page - 1}"))
    if page + 1 < total_pages:
        nav_row.append(InlineKeyboardButton("Siguiente", callback_data=f"page|{picker_id}|{page + 1}"))
    if nav_row:
        rows.append(nav_row)

    if show_back:
        rows.extend(navigation_markup())

    return InlineKeyboardMarkup(rows)


def _strip_text(text: str) -> str:
    return " ".join(text.split())


def _format_picker_title(title: str, subtitle: str, items: List[Dict[str, Any]], page: int = 0) -> str:
    total_pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
    return (
        f"{title}\n"
        f"{subtitle}\n"
        f"Pagina {page + 1}/{total_pages}\n"
        f"Selecciona una opcion:"
    )


def _get_picker_state(context: ContextTypes.DEFAULT_TYPE) -> Dict[str, Any]:
    state = context.user_data.setdefault("picker", {})
    return state


def _set_picker(
    context: ContextTypes.DEFAULT_TYPE,
    *,
    flow: str,
    step: str,
    title: str,
    subtitle: str,
    items: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
    page: int = 0,
) -> None:
    picker = _get_picker_state(context)
    picker.clear()
    picker.update(
        {
            "id": secrets.token_hex(4),
            "flow": flow,
            "step": step,
            "title": title,
            "subtitle": subtitle,
            "items": items,
            "page": page,
            "meta": meta or {},
        }
    )


def _clear_state(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()


async def _show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str = "Menú principal:") -> int:
    _clear_state(context)
    if update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=main_menu_markup())
    else:
        await update.effective_message.reply_text(text, reply_markup=main_menu_markup())
    return ROOT


async def _send_report(bot, chat_id: int, title: str, body: str, reply_markup: Optional[InlineKeyboardMarkup] = None) -> None:
    await bot.send_message(
        chat_id=chat_id,
        text=f"<b>{escape(title)}</b>",
        parse_mode="HTML",
        reply_markup=reply_markup,
        disable_web_page_preview=True,
    )
    for chunk in split_telegram_message(body, limit=REPORT_LIMIT):
        if not chunk.strip():
            continue
        await bot.send_message(
            chat_id=chat_id,
            text=f"<pre>{escape(chunk)}</pre>",
            parse_mode="HTML",
            disable_web_page_preview=True,
        )


async def _send_menu_after_report(bot, chat_id: int) -> None:
    await bot.send_message(chat_id=chat_id, text="Menú principal:", reply_markup=main_menu_markup())


def _build_competition_items() -> List[Dict[str, Any]]:
    return [{"label": comp, "value": comp} for comp in available_competitions()]


def _build_team_items(competition: str) -> List[Dict[str, Any]]:
    teams, _ = get_competition_teams_and_matches(competition)
    return [{"label": team, "value": team} for team in teams]


def _build_match_items_for_flow(stage: str, status: str, days_back: int, days_forward: int) -> List[Dict[str, Any]]:
    matches = load_worldcup_matches(
        stage=stage if stage else None,
        status=status if status else None,
        days_back=days_back,
        days_forward=days_forward,
    )
    if stage in {"round_of_32", "round_of_16"}:
        bracket_matches = _build_bracket_worldcup_matches(stage=stage, status=status)
        if bracket_matches:
            matches = bracket_matches
    if not matches and stage in {"round_of_32", "round_of_16"}:
        matches = _build_local_worldcup_matches(stage=stage, status=status)
    return build_match_items(matches)


@lru_cache(maxsize=1)
def _load_local_team_stats_rows() -> tuple[Dict[str, Any], ...]:
    if not LOCAL_TEAM_STATS_PATH.exists():
        return tuple()
    rows: List[Dict[str, Any]] = []
    with LOCAL_TEAM_STATS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return tuple(rows)


@lru_cache(maxsize=128)
def _load_espn_summary(event_id: str) -> Dict[str, Any]:
    from predicciones.src.data.espn_client import EspnWorldCupClient

    return EspnWorldCupClient(timeout=15).get_summary(event_id)


@lru_cache(maxsize=1)
def _load_bracket_event_ids() -> tuple[str, ...]:
    try:
        response = requests.get(
            WORLD_CUP_BRACKET_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("bracket_fetch_failed error=%s", exc)
        return tuple()

    ids = sorted(set(re.findall(r'/soccer/match/_/gameId/(\d+)/', response.text)))
    return tuple(ids)


def _infer_stage_from_summary(summary: Dict[str, Any]) -> str:
    header = summary.get("header", {}) if isinstance(summary, dict) else {}
    competitions = header.get("competitions", []) if isinstance(header, dict) else []
    if not competitions:
        return ""
    alt_note = str(competitions[0].get("altGameNote") or "").lower()
    if "round of 32" in alt_note:
        return "round_of_32"
    if "round of 16" in alt_note:
        return "round_of_16"
    if "quarterfinal" in alt_note:
        return "quarter_final"
    if "semifinal" in alt_note:
        return "semi_final"
    if "third place" in alt_note:
        return "third_place"
    if "final" in alt_note:
        return "final"
    if "group" in alt_note:
        return "group"
    return ""


def _build_local_worldcup_matches(*, stage: str, status: str) -> List[Dict[str, Any]]:
    rows = _load_local_team_stats_rows()
    filtered: List[Dict[str, Any]] = []
    status_lower = status.lower().strip() if status else ""

    for row in rows:
        event_id = str(row.get("event_id") or "").strip()
        if not event_id or int(event_id) < 760500:
            continue

        summary = _load_espn_summary(event_id)
        if not summary:
            continue

        inferred_stage = _infer_stage_from_summary(summary)
        if stage and inferred_stage != stage:
            continue

        row_status = str(row.get("status") or "").lower().strip()
        if status_lower and status_lower not in {row_status, "post", "completed"}:
            continue

        filtered.append(
            {
                "event_id": event_id,
                "date": str(row.get("date") or ""),
                "competition": str(row.get("competition") or "FIFA World Cup"),
                "stage": inferred_stage or stage,
                "status": "post" if row_status in {"completed", "post"} else row_status,
                "completed": True,
                "neutral_venue": bool(row.get("neutral_venue", False)),
                "venue": row.get("venue"),
                "home_team": str(row.get("home_team") or ""),
                "away_team": str(row.get("away_team") or ""),
                "home_score": row.get("home_score"),
                "away_score": row.get("away_score"),
                "stats": {
                    "home_shots": row.get("home_shots"),
                    "away_shots": row.get("away_shots"),
                    "home_shots_on_target": row.get("home_shots_on_target"),
                    "away_shots_on_target": row.get("away_shots_on_target"),
                    "home_possession": row.get("home_possession"),
                    "away_possession": row.get("away_possession"),
                    "home_corners": row.get("home_corners"),
                    "away_corners": row.get("away_corners"),
                    "home_fouls": row.get("home_fouls"),
                    "away_fouls": row.get("away_fouls"),
                },
            }
        )

    filtered.sort(key=lambda m: (str(m.get("date") or ""), str(m.get("home_team") or ""), str(m.get("away_team") or "")))
    return filtered


def _build_bracket_worldcup_matches(*, stage: str, status: str) -> List[Dict[str, Any]]:
    stage_lower = stage.lower().strip() if stage else ""
    status_lower = status.lower().strip() if status else ""
    matches: List[Dict[str, Any]] = []

    for event_id in _load_bracket_event_ids():
        summary = _load_espn_summary(event_id)
        if not summary:
            continue

        inferred_stage = _infer_stage_from_summary(summary)
        if stage_lower and inferred_stage != stage_lower:
            continue

        header = summary.get("header", {}) if isinstance(summary, dict) else {}
        competitions = header.get("competitions", []) if isinstance(header, dict) else []
        if not competitions:
            continue

        competition = competitions[0]
        status_info = competition.get("status", {})
        status_type = status_info.get("type", {}) if isinstance(status_info, dict) else {}
        normalized_status = str(status_type.get("state") or "").lower()
        if not normalized_status and bool(status_type.get("completed")):
            normalized_status = "post"
        if status_lower and status_lower not in {normalized_status, "post", "completed"}:
            continue

        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = competitors[0]
        away = competitors[1]
        venue_info = summary.get("gameInfo", {}).get("venue", {}) if isinstance(summary, dict) else {}
        venue_name = ""
        if isinstance(venue_info, dict):
            venue_name = venue_info.get("fullName") or venue_info.get("address", {}).get("city") or ""

        matches.append(
            {
                "event_id": str(event_id),
                "date": str(competition.get("date") or header.get("date") or ""),
                "competition": str((header.get("league") or {}).get("name") or "FIFA World Cup"),
                "stage": inferred_stage or stage_lower,
                "status": normalized_status or "post",
                "completed": bool(status_type.get("completed", True)),
                "neutral_venue": bool(competition.get("neutralSite", False)),
                "venue": venue_name,
                "home_team": str(home.get("team", {}).get("displayName") or home.get("team", {}).get("name") or ""),
                "away_team": str(away.get("team", {}).get("displayName") or away.get("team", {}).get("name") or ""),
                "home_score": int(home.get("score")) if str(home.get("score") or "").isdigit() else None,
                "away_score": int(away.get("score")) if str(away.get("score") or "").isdigit() else None,
                "home_winner": bool(home.get("winner")) if home.get("winner") is not None else None,
                "away_winner": bool(away.get("winner")) if away.get("winner") is not None else None,
            }
        )

    matches.sort(key=lambda m: (str(m.get("date") or ""), str(m.get("home_team") or ""), str(m.get("away_team") or "")))
    return matches


def _build_stage_items_for_flow() -> List[Dict[str, Any]]:
    return build_stage_items()


def _build_player_items(team: str) -> List[Dict[str, Any]]:
    rows = get_player_rows_for_team(team)
    return [{"label": row.get("player_name", "N/A"), "value": row.get("player_name", "")} for row in rows if row.get("player_name")]


async def _render_picker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    picker = _get_picker_state(context)
    items = picker.get("items") or []
    page = int(picker.get("page") or 0)
    title = picker.get("title", "Seleccion")
    subtitle = picker.get("subtitle", "")
    keyboard = build_picker_markup(picker["id"], items, page=page)
    text = _format_picker_title(title, subtitle, items, page)
    if update.callback_query:
        await update.callback_query.message.edit_text(text, reply_markup=keyboard)
    else:
        await update.effective_message.reply_text(text, reply_markup=keyboard)
    return PICK


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("command=/start chat=%s", update.effective_chat.id if update.effective_chat else None)
    return await _show_main_menu(update, context, "Bot listo. Selecciona una opcion:")


async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("command=/menu chat=%s", update.effective_chat.id if update.effective_chat else None)
    return await _show_main_menu(update, context, "Menú principal:")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("command=/help chat=%s", update.effective_chat.id if update.effective_chat else None)
    await update.effective_message.reply_text(
        "Comandos:\n"
        "/start - reinicia el flujo\n"
        "/menu - abre el menú principal\n"
        "/help - esta ayuda\n"
        "/cancel - cancela el flujo actual\n\n"
        "El bot usa estado en memoria por chat; si se reinicia el proceso o expira la conversación, vuelve a usar /start.\n"
        "Prediccion, Parlay, Timeline y Analisis se navegan por fase del torneo y luego por partido.",
        reply_markup=main_menu_markup(),
    )
    return ROOT


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("command=/cancel chat=%s", update.effective_chat.id if update.effective_chat else None)
    return await _show_main_menu(update, context, "Flujo cancelado.")


async def timeout_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("conversation_timeout chat=%s", update.effective_chat.id if update.effective_chat else None)
    return await _show_main_menu(update, context, "La conversacion expiró. Se reinició el menú.")


async def unexpected_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("unexpected_text chat=%s text=%s", update.effective_chat.id if update.effective_chat else None, _strip_text(update.effective_message.text or ""))
    await update.effective_message.reply_text("Usa los botones del menú o /menu para reiniciar.", reply_markup=main_menu_markup())
    return PICK


async def _handle_main_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE, choice: str) -> int:
    logger.info("option=%s chat=%s", choice, update.effective_chat.id if update.effective_chat else None)
    _clear_state(context)
    if choice in {"prediction", "parlay", "timelines", "analysis"}:
        flow_titles = {
            "prediction": "🔮 Prediccion",
            "parlay": "🧩 Generar Parlay",
            "timelines": "⏱️ Time lines de partidos",
            "analysis": "📊 Analisis de Partidos",
        }
        flow_subtitles = {
            "prediction": "Selecciona la fase del torneo:",
            "parlay": "Selecciona la fase del torneo:",
            "timelines": "Selecciona la fase del torneo:",
            "analysis": "Selecciona la fase del torneo:",
        }
        _set_picker(
            context,
            flow=choice,
            step="stage",
            title=flow_titles[choice],
            subtitle=flow_subtitles[choice],
            items=build_stage_items(),
            meta={
                "status": "pre" if choice in {"prediction", "parlay"} else "post",
                "days_back": 0 if choice in {"prediction", "parlay"} else 365,
                "days_forward": 7 if choice in {"prediction", "parlay"} else 0,
            },
        )
    elif choice == "players":
        competitions = _build_competition_items()
        if not competitions:
            return await _show_main_menu(update, context, "No encontré competiciones disponibles.")
        _set_picker(
            context,
            flow="players",
            step="competition",
            title="👤 Datos de jugadores",
            subtitle="Selecciona una liga/competicion:",
            items=competitions,
        )
    else:
        return await _show_main_menu(update, context, "Menú principal:")
    return await _render_picker(update, context)


def _resolve_picker_item(context: ContextTypes.DEFAULT_TYPE, index: int) -> Optional[Dict[str, Any]]:
    picker = _get_picker_state(context)
    items = picker.get("items") or []
    if 0 <= index < len(items):
        return items[index]
    return None


async def _go_back(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    picker = _get_picker_state(context)
    flow = picker.get("flow")
    step = picker.get("step")

    if flow and step == "stage":
        return await _show_main_menu(update, context, "Menú principal:")

    if flow in {"prediction", "parlay", "timelines", "analysis"} and step == "match":
        meta = picker.get("meta", {})
        _set_picker(
            context,
            flow=flow,
            step="stage",
            title=picker.get("title", "Seleccion"),
            subtitle="Selecciona la fase del torneo:",
            items=_build_stage_items_for_flow(),
            meta=meta,
        )
        return await _render_picker(update, context)

    if flow in {"prediction", "parlay", "timelines", "players"} and step in {"team", "player", "confirm"}:
        competition = picker.get("meta", {}).get("competition", "")
        if not competition:
            return await _show_main_menu(update, context, "Menú principal:")
        _set_picker(
            context,
            flow=flow,
            step="competition",
            title=picker.get("title", "Seleccion"),
            subtitle="Selecciona una liga/competicion:",
            items=_build_competition_items(),
        )
        return await _render_picker(update, context)

    return await _show_main_menu(update, context, "Menú principal:")


async def _handle_pick(update: Update, context: ContextTypes.DEFAULT_TYPE, index: int) -> int:
    picker = _get_picker_state(context)
    item = _resolve_picker_item(context, index)
    if item is None:
        await update.callback_query.answer("Seleccion invalida o expirada.", show_alert=False)
        return PICK

    flow = picker.get("flow")
    step = picker.get("step")
    meta = dict(picker.get("meta") or {})
    logger.info("pick flow=%s step=%s index=%s chat=%s", flow, step, index, update.effective_chat.id if update.effective_chat else None)

    if flow in {"prediction", "parlay", "timelines", "analysis"} and step == "stage":
        stage = item["value"]
        status = str(meta.get("status") or "")
        days_back = int(meta.get("days_back") or 0)
        days_forward = int(meta.get("days_forward") or 0)
        matches = await asyncio.to_thread(
            _build_match_items_for_flow,
            stage,
            status,
            days_back,
            days_forward,
        )
        if not matches:
            await update.callback_query.message.reply_text(
                "No encontré partidos para esa fase en la ventana actual. Prueba otra fase.",
            )
            return PICK
        _set_picker(
            context,
            flow=flow,
            step="match",
            title={
                "prediction": "🔮 Prediccion",
                "parlay": "🧩 Generar Parlay",
                "timelines": "⏱️ Time lines de partidos",
                "analysis": "📊 Analisis de Partidos",
            }[flow],
            subtitle=f"Fase: {_stage_title(stage)}\nSelecciona un partido:",
            items=matches,
            meta={
                **meta,
                "stage": stage,
                "matches": matches,
            },
        )
        return await _render_picker(update, context)

    if flow == "prediction" and step == "match":
        match = item.get("match") or {}
        home = str(match.get("home_team") or "")
        away = str(match.get("away_team") or "")
        competition = str(match.get("competition") or "FIFA World Cup")
        await update.callback_query.message.edit_text("Generando prediccion...")
        try:
            response = await asyncio.to_thread(
                predict_match_pipeline,
                home,
                away,
                match.get("date"),
                bool(match.get("neutral_venue", False)),
                True,
                competition,
                "fifa.world",
            )
            text = format_prediction_text(response)
            await _send_report(context.bot, update.effective_chat.id, "Prediccion", text)
        except Exception as exc:
            logger.exception("prediction_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando la prediccion: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT

    if flow == "parlay" and step == "match":
        match = item.get("match") or {}
        home = str(match.get("home_team") or "")
        away = str(match.get("away_team") or "")
        competition = str(match.get("competition") or "FIFA World Cup")
        await update.callback_query.message.edit_text("Generando parlay...")
        try:
            pred = await asyncio.to_thread(
                predict_match_pipeline,
                home,
                away,
                match.get("date"),
                bool(match.get("neutral_venue", False)),
                True,
                competition,
                "fifa.world",
            )
            calib_manager = await asyncio.to_thread(load_calibration_manager)
            calib_status = check_calibration()
            parlays, _ = await asyncio.to_thread(
                build_all_same_game_parlays,
                pred,
                home,
                away,
                calib_status,
                calib_manager if calib_manager.calibrators else None,
            )
            text = format_parlay_text(home, away, pred, parlays)
            await _send_report(context.bot, update.effective_chat.id, "Generar Parlay", text)
        except Exception as exc:
            logger.exception("parlay_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando el parlay: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT

    if flow == "timelines" and step == "match":
        match = item.get("match") or {}
        await update.callback_query.message.edit_text("Generando timeline...")
        try:
            text = await asyncio.to_thread(format_timeline_text, "", match)
            await _send_report(context.bot, update.effective_chat.id, "Time lines de partidos", text)
        except Exception as exc:
            logger.exception("timeline_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando timeline: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT

    if flow == "analysis" and step == "match":
        match = item.get("match") or {}
        event_id = str(match.get("event_id") or "")
        if not event_id:
            await update.callback_query.message.reply_text("No pude recuperar el evento del partido.")
            return await _show_main_menu(update, context, "Volviendo al menú principal.")
        await update.callback_query.message.edit_text("Generando análisis...")
        try:
            from predicciones.src.data.espn_client import EspnWorldCupClient

            summary = await asyncio.to_thread(EspnWorldCupClient(timeout=15).get_summary, event_id)
            if not summary:
                raise RuntimeError("ESPN no devolvió resumen para este partido")
            text = await asyncio.to_thread(format_match_analysis_text, match, summary)
            await _send_report(context.bot, update.effective_chat.id, "Analisis de Partidos", text)
        except Exception as exc:
            logger.exception("analysis_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando el analisis: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT

    if flow == "players" and step == "competition":
        competition = item["value"]
        teams = _build_team_items(competition)
        if not teams:
            await update.callback_query.message.reply_text("No encontré equipos para esa competicion.")
            return await _show_main_menu(update, context, "Volviendo al menú principal.")
        _set_picker(
            context,
            flow="players",
            step="team",
            title="👤 Datos de jugadores",
            subtitle=f"Liga: {competition}\nSelecciona el equipo:",
            items=teams,
            meta={"competition": competition},
        )
        return await _render_picker(update, context)

    if flow == "players" and step == "team":
        competition = meta.get("competition", "")
        team = item["value"]
        players = _build_player_items(team)
        if not players:
            await update.callback_query.message.reply_text("No se encontraron jugadores para ese equipo.")
            await _send_menu_after_report(context.bot, update.effective_chat.id)
            return ROOT
        _set_picker(
            context,
            flow="players",
            step="player",
            title="👤 Datos de jugadores",
            subtitle=f"{competition}\nEquipo: {team}\nSelecciona el jugador:",
            items=players,
            meta={"competition": competition, "team": team},
        )
        return await _render_picker(update, context)

    if flow == "players" and step == "player":
        competition = meta.get("competition", "")
        team = meta.get("team", "")
        player_name = item["value"]
        await update.callback_query.message.edit_text("Generando datos de jugadores...")
        try:
            text = await asyncio.to_thread(format_player_detail, team, player_name)
            await _send_report(context.bot, update.effective_chat.id, "Datos de jugadores", text)
        except Exception as exc:
            logger.exception("players_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando datos de jugadores: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT

    await update.callback_query.answer("No pude procesar esa opcion.", show_alert=False)
    return PICK


async def _handle_action(update: Update, context: ContextTypes.DEFAULT_TYPE, action: str) -> int:
    logger.info("action=%s chat=%s", action, update.effective_chat.id if update.effective_chat else None)
    if action == "confirm":
        picker = _get_picker_state(context)
        meta = picker.get("meta") or {}
        competition = meta.get("competition", "")
        home = meta.get("home_team", "")
        away = meta.get("away_team", "")
        if not home or not away:
            return await _show_main_menu(update, context, "No pude recuperar el partido. Reiniciando.")
        await update.callback_query.message.edit_text("Generando prediccion...")
        try:
            response = await asyncio.to_thread(
                predict_match_pipeline,
                home,
                away,
                None,
                False,
                True,
                competition,
                "fifa.world",
            )
            text = format_prediction_text(response)
            await _send_report(context.bot, update.effective_chat.id, "Prediccion", text)
        except Exception as exc:
            logger.exception("prediction_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando la prediccion: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT
    return await _show_main_menu(update, context, "Menú principal:")


async def callback_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data = query.data or ""

    if data.startswith("menu|"):
        choice = data.split("|", 1)[1]
        return await _handle_main_menu_choice(update, context, choice)

    if data.startswith("page|"):
        _, picker_id, page_raw = data.split("|", 2)
        picker = _get_picker_state(context)
        if picker.get("id") != picker_id:
            await query.message.reply_text("Ese menu ya expiró. Usa /menu para reiniciar.")
            return ROOT
        picker["page"] = int(page_raw)
        return await _render_picker(update, context)

    if data.startswith("pick|"):
        _, picker_id, idx_raw = data.split("|", 2)
        picker = _get_picker_state(context)
        if picker.get("id") != picker_id:
            await query.message.reply_text("Ese menu ya expiró. Usa /menu para reiniciar.")
            return ROOT
        return await _handle_pick(update, context, int(idx_raw))

    if data.startswith("act|"):
        return await _handle_action(update, context, data.split("|", 1)[1])

    if data.startswith("nav|"):
        action = data.split("|", 1)[1]
        if action == "back":
            return await _go_back(update, context)
        if action == "menu":
            return await _show_main_menu(update, context, "Menú principal:")
        if action == "cancel":
            return await _show_main_menu(update, context, "Flujo cancelado.")

    await query.message.reply_text("No pude procesar esa accion. Usa /menu para reiniciar.")
    return ROOT


def build_application():
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("BOT_TOKEN is required")

    application = ApplicationBuilder().token(token).build()
    application.add_error_handler(_log_error)

    conversation = ConversationHandler(
        entry_points=[
            CommandHandler("start", start_command),
            CommandHandler("menu", menu_command),
        ],
        states={
            ROOT: [
                CallbackQueryHandler(callback_router),
                MessageHandler(filters.TEXT & ~filters.COMMAND, unexpected_text),
            ],
            PICK: [
                CallbackQueryHandler(callback_router),
                MessageHandler(filters.TEXT & ~filters.COMMAND, unexpected_text),
            ],
            ConversationHandler.TIMEOUT: [MessageHandler(filters.ALL, timeout_handler)],
        },
        fallbacks=[
            CommandHandler("start", start_command),
            CommandHandler("menu", menu_command),
            CommandHandler("help", help_command),
            CommandHandler("cancel", cancel_command),
        ],
        allow_reentry=True,
        conversation_timeout=900,
    )

    application.add_handler(conversation)
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    return application


async def _log_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("telegram_error update=%s", update, exc_info=context.error)


def main() -> None:
    application = build_application()
    logger.info("Starting Telegram bot with polling")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
