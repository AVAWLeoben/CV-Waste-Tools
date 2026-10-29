"""Detection augmentation overrides and a compact, transactional Tk editor."""
import math

# key: (label, minimum, maximum, suggested increment)
FIELDS = {
    'mosaic': ('Mosaic probability', 0, 1, .1),
    'mixup': ('MixUp probability', 0, 1, .05),
    'degrees': ('Rotation (degrees)', 0, 180, 5),
    'translate': ('Translation fraction', 0, 1, .05),
    'scale': ('Scale variation', 0, 1, .05),
    'fliplr': ('Horizontal flip probability', 0, 1, .1),
    'flipud': ('Vertical flip probability', 0, 1, .1),
    'hsv_h': ('Hue variation', 0, 1, .005),
    'hsv_s': ('Saturation variation', 0, 1, .05),
    'hsv_v': ('Brightness variation', 0, 1, .05),
    'close_mosaic': ('Disable mosaic for final epochs', 0, 10000, 1),
}
GENTLE = dict(mosaic=.3, mixup=0., degrees=10., translate=.1, scale=.25,
              fliplr=.5, flipud=0., hsv_h=.015, hsv_s=.3, hsv_v=.25,
              close_mosaic=10)


def validate_augmentation(values):
    """Blank/missing keys inherit the installed Ultralytics defaults."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError('Augmentation must be a settings object.')
    result = {}
    for key, value in values.items():
        if key not in FIELDS:
            raise ValueError(f'Unknown augmentation option: {key}')
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        label, low, high, _ = FIELDS[key]
        if isinstance(value, bool):
            raise ValueError(f'{label}: enter a number, not a boolean.')
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{label}: enter a number.') from None
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f'{label}: use a value from {low} to {high}.')
        if key == 'close_mosaic':
            if not number.is_integer():
                raise ValueError('Final mosaic epochs must be a whole number.')
            number = int(number)
        result[key] = number
    return result


def edit_augmentation(parent, current, on_apply):
    import tkinter as tk
    from tkinter import ttk, messagebox
    dialog = tk.Toplevel(parent)
    dialog.title('Training augmentation')
    dialog.transient(parent.winfo_toplevel())
    dialog.resizable(True, True)
    frame = ttk.Frame(dialog, padding=16)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Augmentation', font=('', 14, 'bold')).pack(anchor='w')
    ttk.Label(frame, text='Blank fields use your installed Ultralytics defaults.\nChanges apply only after clicking Apply.',
              wraplength=470).pack(anchor='w', pady=(4, 12))
    preset = tk.StringVar(value='Custom' if current else 'Library defaults')
    selector = ttk.Combobox(frame, state='readonly', textvariable=preset,
                           values=['Library defaults', 'Gentle handheld', 'Custom'])
    selector.pack(fill='x', pady=(0, 10))
    book = ttk.Notebook(frame); book.pack(fill='both', expand=True)
    variables = {}
    for title, keys in [('Geometry', ['degrees','translate','scale','fliplr','flipud']),
                        ('Colour', ['hsv_h','hsv_s','hsv_v']),
                        ('Image mixing', ['mosaic','mixup','close_mosaic'])]:
        page = ttk.Frame(book, padding=12);book.add(page, text=title)
        page.columnconfigure(0, weight=1)
        for row, key in enumerate(keys):
            label, low, high, increment = FIELDS[key]
            ttk.Label(page, text=f'{label} ({low}–{high})').grid(row=row,column=0,sticky='w',pady=6)
            var=tk.StringVar(value=str(current.get(key,'')));variables[key]=var
            ttk.Spinbox(page,from_=low,to=high,increment=increment,textvariable=var,width=10).grid(row=row,column=1,padx=(12,0),pady=6)
            var.trace_add('write', lambda *_: preset.set('Custom'))
    def choose(_=None):
        name=preset.get()
        if name=='Custom':return
        values=GENTLE if name=='Gentle handheld' else {}
        for key,var in variables.items():var.set(str(values.get(key,'')))
        preset.set(name)
    selector.bind('<<ComboboxSelected>>',choose)
    ttk.Label(frame,text='Gentle handheld is an experimental preset for single objects held in hand.\nColour controls can affect clear/coloured glass recognition.',wraplength=470).pack(anchor='w',pady=12)
    buttons=ttk.Frame(frame);buttons.pack(fill='x')
    def apply():
        try: result=validate_augmentation({k:v.get() for k,v in variables.items()})
        except ValueError as exc:
            messagebox.showerror('Invalid augmentation',str(exc),parent=dialog);return
        on_apply(result);dialog.destroy()
    ttk.Button(buttons,text='Cancel',command=dialog.destroy).pack(side='right')
    ttk.Button(buttons,text='Apply',command=apply).pack(side='right',padx=8)
    dialog.bind('<Escape>',lambda _:dialog.destroy())
    dialog.grab_set();selector.focus_set()
    return dialog
