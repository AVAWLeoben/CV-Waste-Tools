AVAW LiteRT Export Menu
=======================

Copy these two files to:

C:\Users\GKoinig\Documents\WSL

  export_litert_menu.bat
  export_to_litert_menu_cli.py

Place your .pt models in the same folder.

Double-click export_litert_menu.bat.

The menu will:
1. Scan the folder for .pt files.
2. Let you select a model by number.
3. Let you choose:
   - 160
   - 320
   - 640
   - Custom
4. Optionally change:
   - NMS
   - INT8
   - HALF
5. Start WSL.
6. Activate ~/yolo_export.
7. Export the selected model to LiteRT.

Defaults:
  format = litert
  nms    = none
  int8   = false
  half   = false

Custom image size examples:
  512
  640,384
