#!/usr/bin/env bash
# CV Waste Studio: create Linux application-menu and desktop shortcuts.
# Place this script next to START_GREENPLAST.sh in CV_Waste_Studio.
set -Eeuo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LAUNCHER="$APP_DIR/START_GREENPLAST.sh"

if [[ ! -f "$LAUNCHER" ]]; then
  printf 'ERROR: The launcher was not found at:\n  %s\n' "$LAUNCHER" >&2
  printf 'Put this installer in the CV_Waste_Studio folder alongside START_GREENPLAST.sh.\n' >&2
  exit 1
fi

# Look for the actual supplied logo, regardless of PNG resolution.
ICON=""
for filename in cv_waste_studio_256.png cv_waste_studio.png \
                cv_waste_studio_128.png cv_waste_studio_64.png \
                cv_waste_studio_sidebar.png cv_waste_studio_48.png \
                cv_waste_studio_32.png; do
  if [[ -s "$APP_DIR/assets/$filename" ]]; then
    ICON="$APP_DIR/assets/$filename"
    break
  fi
done
if [[ -z "$ICON" ]]; then
  ICON="applications-graphics"
  printf 'Warning: no PNG icon found in assets/. Using a system fallback icon.\n' >&2
fi

# Escape an argument according to freedesktop.org Desktop Entry's Exec rules.
desktop_exec_quote() {
  local value="$1"
  value="${value//%/%%}"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//\$/\\\$}"
  value="${value//\`/\\\`}"
  printf '"%s"' "$value"
}

APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p -- "$APPLICATIONS_DIR"
ENTRY="$APPLICATIONS_DIR/cv-waste-studio.desktop"

cat > "$ENTRY" <<DESKTOP_ENTRY
[Desktop Entry]
Version=1.0
Type=Application
Name=CV Waste Studio
Comment=Dataset preparation, model training and evaluation
Exec=/usr/bin/env bash $(desktop_exec_quote "$LAUNCHER")
Path=$APP_DIR
Icon=$ICON
Terminal=false
Categories=Development;Graphics;
StartupNotify=true
DESKTOP_ENTRY
chmod 755 -- "$ENTRY"
printf 'Application menu shortcut: %s\n' "$ENTRY"

DESKTOP_DIR=""
if command -v xdg-user-dir >/dev/null 2>&1; then
  DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
fi
if [[ -z "$DESKTOP_DIR" || "$DESKTOP_DIR" == "$HOME" ]]; then
  DESKTOP_DIR="$HOME/Desktop"
fi
if [[ -d "$DESKTOP_DIR" ]]; then
  SHORTCUT="$DESKTOP_DIR/CV Waste Studio.desktop"
  cp -- "$ENTRY" "$SHORTCUT"
  chmod 755 -- "$SHORTCUT"
  if command -v gio >/dev/null 2>&1; then
    gio set "$SHORTCUT" metadata::trusted true >/dev/null 2>&1 || true
  fi
  printf 'Desktop shortcut: %s\n' "$SHORTCUT"
else
  printf 'No desktop directory found (%s); application-menu entry installed.\n' "$DESKTOP_DIR"
fi

printf 'Icon: %s\n' "$ICON"
printf 'To start without a shortcut, run: bash "%s"\n' "$LAUNCHER"
