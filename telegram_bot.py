from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from predicciones.src.models.parlay_builder import RiskLevel, build_all_same_game_parlays, check_calibration  # noqa: E402
from predicciones.src.pipeline.predict import predict_match_pipeline  # noqa: E402
from predicciones.src.telegram_integration import (  # noqa: E402
    available_competitions,
    available_team_names,
    format_parlay_text,
    format_player_detail,
    format_player_team_summary,
    format_prediction_text,
    format_timeline_text,
    get_competition_teams_and_matches,
    get_matches_for_competition,
    get_player_rows_for_team,
    get_team_matches,
    load_calibration_manager,
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


def main_menu_markup() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("Prediccion", callback_data="menu|prediction")],
            [InlineKeyboardButton("Generar Parlay", callback_data="menu|parlay")],
            [InlineKeyboardButton("Time lines de partidos", callback_data="menu|timelines")],
            [InlineKeyboardButton("Datos de jugadores", callback_data="menu|players")],
        ]
    )


def navigation_markup() -> List[List[InlineKeyboardButton]]:
    return [
        [
            InlineKeyboardButton("Volver", callback_data="nav|back"),
            InlineKeyboardButton("Menu principal", callback_data="nav|menu"),
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


def _build_match_items(competition: str) -> List[Dict[str, Any]]:
    matches = get_matches_for_competition(competition)
    items: List[Dict[str, Any]] = []
    for idx, match in enumerate(matches):
        label = f"{match.home_team} vs {match.away_team}"
        if match.date:
            label += f" | {match.date[:10]}"
        items.append({"label": label, "value": idx, "match": match})
    return items


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
        "El bot usa estado en memoria por chat; si se reinicia el proceso o expira la conversación, vuelve a usar /start.",
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
    picker = _get_picker_state(context)
    if choice == "prediction":
        competitions = _build_competition_items()
        if not competitions:
            return await _show_main_menu(update, context, "No encontré competiciones disponibles.")
        _set_picker(
            context,
            flow="prediction",
            step="competition",
            title="Prediccion",
            subtitle="Selecciona una liga/competicion:",
            items=competitions,
        )
    elif choice == "parlay":
        competitions = _build_competition_items()
        if not competitions:
            return await _show_main_menu(update, context, "No encontré competiciones disponibles.")
        _set_picker(
            context,
            flow="parlay",
            step="competition",
            title="Generar Parlay",
            subtitle="Selecciona una liga/competicion:",
            items=competitions,
        )
    elif choice == "timelines":
        competitions = _build_competition_items()
        if not competitions:
            return await _show_main_menu(update, context, "No encontré competiciones disponibles.")
        _set_picker(
            context,
            flow="timelines",
            step="competition",
            title="Time lines de partidos",
            subtitle="Selecciona una liga/competicion:",
            items=competitions,
        )
    elif choice == "players":
        competitions = _build_competition_items()
        if not competitions:
            return await _show_main_menu(update, context, "No encontré competiciones disponibles.")
        _set_picker(
            context,
            flow="players",
            step="competition",
            title="Datos de jugadores",
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

    if flow and step == "competition":
        return await _show_main_menu(update, context, "Menú principal:")

    if flow in {"prediction", "parlay", "timelines", "players"} and step in {"team", "match", "player", "risk", "confirm"}:
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

    if flow == "prediction" and step == "competition":
        competition = item["value"]
        teams = _build_team_items(competition)
        if not teams:
            await update.callback_query.message.reply_text("No encontré equipos para esa competicion.")
            return await _show_main_menu(update, context, "Volviendo al menú principal.")
        _set_picker(
            context,
            flow="prediction",
            step="home",
            title="Prediccion",
            subtitle=f"Liga: {competition}\nSelecciona el equipo local:",
            items=teams,
            meta={"competition": competition},
        )
        return await _render_picker(update, context)

    if flow == "prediction" and step == "home":
        competition = meta.get("competition", "")
        home = item["value"]
        teams = [team for team in _build_team_items(competition) if team["value"] != home]
        _set_picker(
            context,
            flow="prediction",
            step="away",
            title="Prediccion",
            subtitle=f"Liga: {competition}\nLocal: {home}\nSelecciona el visitante:",
            items=teams,
            meta={"competition": competition, "home_team": home},
        )
        return await _render_picker(update, context)

    if flow == "prediction" and step == "away":
        competition = meta.get("competition", "")
        home = meta.get("home_team", "")
        away = item["value"]
        _set_picker(
            context,
            flow="prediction",
            step="confirm",
            title="Prediccion",
            subtitle=f"Confirmar:\n{competition}\n{home} vs {away}",
            items=[{"label": "Confirmar prediccion", "value": "confirm"}],
            meta={"competition": competition, "home_team": home, "away_team": away},
        )
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("Confirmar", callback_data="act|confirm")],
                [InlineKeyboardButton("Volver", callback_data="nav|back"), InlineKeyboardButton("Menu principal", callback_data="nav|menu")],
                [InlineKeyboardButton("Cancelar", callback_data="nav|cancel")],
            ]
        )
        if update.callback_query:
            await update.callback_query.message.edit_text(
                _format_picker_title("Prediccion", f"Confirmar:\n{competition}\n{home} vs {away}", [], 0),
                reply_markup=keyboard,
            )
        return PICK

    if flow == "parlay" and step == "competition":
        competition = item["value"]
        matches = _build_match_items(competition)
        if not matches:
            await update.callback_query.message.reply_text("No encontré partidos para esa competicion.")
            return await _show_main_menu(update, context, "Volviendo al menú principal.")
        _set_picker(
            context,
            flow="parlay",
            step="match",
            title="Generar Parlay",
            subtitle=f"Liga: {competition}\nSelecciona el partido:",
            items=matches,
            meta={"competition": competition},
        )
        return await _render_picker(update, context)

    if flow == "parlay" and step == "match":
        competition = meta.get("competition", "")
        match = get_matches_for_competition(competition)[index]
        _set_picker(
            context,
            flow="parlay",
            step="risk",
            title="Generar Parlay",
            subtitle=f"{match.home_team} vs {match.away_team}\nElige el riesgo:",
            items=[
                {"label": "Bajo", "value": "low"},
                {"label": "Medio", "value": "medium"},
                {"label": "Alto", "value": "high"},
                {"label": "Ver todo", "value": "all"},
            ],
            meta={
                "competition": competition,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "match_date": match.date or None,
                "neutral_venue": match.neutral_venue,
            },
        )
        return await _render_picker(update, context)

    if flow == "parlay" and step == "risk":
        competition = meta.get("competition", "")
        home = meta.get("home_team", "")
        away = meta.get("away_team", "")
        selected_risk = item["value"]
        await update.callback_query.message.edit_text("Generando parlay...")
        try:
            pred = await asyncio.to_thread(
                predict_match_pipeline,
                home,
                away,
                meta.get("match_date"),
                meta.get("neutral_venue", False),
                False,
                competition,
                competition,
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
            text = format_parlay_text(home, away, pred, parlays, selected_risk if selected_risk != "all" else None)
            await _send_report(context.bot, update.effective_chat.id, "Generar Parlay", text)
        except Exception as exc:
            logger.exception("parlay_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando el parlay: {exc}")
        await _send_menu_after_report(context.bot, update.effective_chat.id)
        return ROOT

    if flow == "timelines" and step == "competition":
        competition = item["value"]
        teams = _build_team_items(competition)
        if not teams:
            await update.callback_query.message.reply_text("No encontré equipos para esa competicion.")
            return await _show_main_menu(update, context, "Volviendo al menú principal.")
        _set_picker(
            context,
            flow="timelines",
            step="team",
            title="Time lines de partidos",
            subtitle=f"Liga: {competition}\nSelecciona el equipo:",
            items=teams,
            meta={"competition": competition},
        )
        return await _render_picker(update, context)

    if flow == "timelines" and step == "team":
        competition = meta.get("competition", "")
        team = item["value"]
        matches = [
            m for m in get_team_matches(team)
            if competition.lower() in str(m.get("competition") or "").lower()
        ]
        match_items = []
        for idx, match in enumerate(matches):
            label = f"{match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}"
            if match.get("date"):
                label += f" | {str(match.get('date'))[:10]}"
            match_items.append({"label": label, "value": idx})
        _set_picker(
            context,
            flow="timelines",
            step="match",
            title="Time lines de partidos",
            subtitle=f"{competition}\nEquipo: {team}\nSelecciona el partido:",
            items=match_items,
            meta={"competition": competition, "team": team, "matches": matches},
        )
        return await _render_picker(update, context)

    if flow == "timelines" and step == "match":
        matches = meta.get("matches") or []
        if index >= len(matches):
            await update.callback_query.answer("Partido invalido.", show_alert=False)
            return PICK
        match = matches[index]
        await update.callback_query.message.edit_text("Generando timeline...")
        try:
            text = await asyncio.to_thread(format_timeline_text, meta.get("team", ""), match)
            await _send_report(context.bot, update.effective_chat.id, "Time lines de partidos", text)
        except Exception as exc:
            logger.exception("timeline_error chat=%s", update.effective_chat.id if update.effective_chat else None)
            await update.callback_query.message.reply_text(f"Error generando timeline: {exc}")
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
            title="Datos de jugadores",
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
            title="Datos de jugadores",
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
                False,
                competition,
                competition,
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
