"""Interactive, dependency-free Tk graphs for Ultralytics native tuning results."""
from __future__ import annotations
import math
import time
import tkinter as tk

COLORS = ('#4269DF', '#ED8C42', '#209980', '#8750C5')
BG = '#F9FBFE'
BORDER = '#DCE5F2'


def ticks(lower, upper, count=5, integers=False):
    if lower == upper:
        return [int(lower)] if integers else [lower]
    if integers:
        gap = max(1, math.ceil((upper - lower) / max(1, count-1)))
        vals = list(range(int(lower), int(upper)+1, gap))
        return vals + ([int(upper)] if vals[-1] != upper else [])
    return [lower + (upper-lower)*j/(count-1) for j in range(count)]


class TuningCharts(tk.Canvas):
    """Four charts; click to enlarge with blue hover affordance and real trial dots."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, highlightbackground=BORDER,
                         highlightthickness=1, bd=0, **kw)
        self.trials = []
        self.parameter = 'lr0'
        self.expanded = None
        self.hover = None
        self.inspected = None
        self.animation = None
        self.bind('<Configure>', lambda _event: self.redraw())
        self.bind('<Motion>', self._motion)
        self.bind('<Leave>', self._leave)
        self.bind('<Button-1>', self._click)
        self.bind('<Escape>', lambda _event: self.collapse())

    def update_trials(self, trials):
        self.trials = trials
        self.redraw()

    def set_parameter(self, key):
        self.parameter = key
        self.redraw()

    def _rect(self, index):
        w,h = self.winfo_width(),self.winfo_height()
        return ((index%2)*w/2, (index//2)*h/2, w/2,h/2)

    def _motion(self, event):
        if self.animation is not None:
            return
        if self.expanded is None:
            w,h = self.winfo_width(),self.winfo_height()
            value = int(event.x>=w/2)+2*int(event.y>=h/2) if 0<=event.x<w and 0<=event.y<h else None
            if value != self.hover:
                self.hover=value
                self.configure(cursor='hand2' if value is not None else '')
                self.redraw()
        else:
            w = self.winfo_width()
            if event.x > w-47 and event.y < 43:
                self.configure(cursor='hand2')
                self.inspected=None
            else:
                self.configure(cursor='crosshair')
                # tooltip snaps to the nearest real trial, not an interpolated value
                if self.trials:
                    self.inspected=min(self.trials, key=lambda p: abs(p['iteration']-
                        (self.trials[0]['iteration']+max(0,(event.x-56))/max(1,w-85)*
                         (self.trials[-1]['iteration']-self.trials[0]['iteration']))))
            self.redraw()

    def _leave(self, _event):
        self.hover=None
        self.inspected=None
        self.configure(cursor='')
        self.redraw()

    def _click(self, event):
        if self.animation is not None:
            return
        self.focus_set()
        if self.expanded is None:
            w,h=self.winfo_width(),self.winfo_height()
            if 0<=event.x<w and 0<=event.y<h:
                self._animate(int(event.x>=w/2)+2*int(event.y>=h/2), True)
        elif event.x > self.winfo_width()-47 and event.y < 43:
            self.collapse()

    def collapse(self):
        if self.expanded is not None and self.animation is None:
            self._animate(self.expanded, False)

    def _animate(self, index, entering):
        self.animation=(index,entering,time.monotonic())
        self._frame()

    def _frame(self):
        if self.animation is None:
            return
        index, entering, start = self.animation
        t=min(1,(time.monotonic()-start)/0.2)
        eased=t*t*(3-2*t)
        if not entering:
            eased=1-eased
        if t>=1:
            self.expanded=index if entering else None
            self.animation=None
            self.redraw()
            return
        self.redraw(transition=(index,eased))
        self.after(16,self._frame)

    def redraw(self, transition=None):
        self.delete('all')
        w,h=self.winfo_width(),self.winfo_height()
        if w<170 or h<140:
            return
        if transition is not None:
            focus, amount = transition
            for j in range(4):
                if j != focus:
                    self._plot(j, *self._rect(j))
            x,y,ww,hh=self._rect(focus)
            self._plot(focus,x*(1-amount),y*(1-amount),ww+(w-ww)*amount,hh+(h-hh)*amount)
        elif self.expanded is not None:
            self._plot(self.expanded,0,0,w,h,enlarged=True)
            self.create_rectangle(w-39,10,w-11,36,fill='#EDF4FF',outline='#AACAF7')
            self.create_text(w-25,23,text='×',fill='#2557A2',font=('Segoe UI',16,'bold'))
            if self.inspected is not None:
                p=self.inspected
                details=f"Trial {p['iteration']}  |  Fitness {p['fitness']:.4f}"
                if self.parameter in p.get('hyperparameters',{}):
                    details+=f"  |  {self.parameter}={p['hyperparameters'][self.parameter]:.5g}"
                self.create_rectangle(max(5,w-365),h-43,w-10,h-12,fill='#EDF4FF',outline='#99B9E9')
                self.create_text(w-20,h-28,text=details,anchor='e',fill='#224B90',font=('Segoe UI',9))
        else:
            for j in range(4):
                self._plot(j,*self._rect(j),hovered=j==self.hover)

    @staticmethod
    def _measure(row,key):
        for candidate in (key, key.replace('(B)','')):
            val=row.get('metrics',{}).get(candidate)
            if val is not None:
                try:
                    f=float(val)
                    return f if math.isfinite(f) else None
                except (ValueError,TypeError):
                    pass
        return None

    def _lines(self,index):
        trials=self.trials
        if index==0:
            top=-math.inf
            best=[]
            for p in trials:
                if not p['failed']:
                    top=max(top,p['fitness'])
                if top>-math.inf:
                    best.append((p['iteration'],top))
            return 'Fitness by trial', [('Trial fitness',COLORS[0],[(p['iteration'],p['fitness']) for p in trials if not p['failed']]),
                                          ('Best so far',COLORS[2],best)],True
        if index==1:
            return 'Validation performance',[
                ('mAP50–95',COLORS[0],[(p['iteration'],v) for p in trials if (v:=self._measure(p,'metrics/mAP50-95(B)')) is not None]),
                ('mAP50',COLORS[1],[(p['iteration'],v) for p in trials if (v:=self._measure(p,'metrics/mAP50(B)')) is not None]),
                ('Recall',COLORS[2],[(p['iteration'],v) for p in trials if (v:=self._measure(p,'metrics/recall(B)')) is not None]),
            ],True
        values=[(p['iteration'],float(p['hyperparameters'][self.parameter])) for p in trials
                if self.parameter in p.get('hyperparameters',{}) and not p['failed']]
        if index==2:
            return f'{self.parameter} · trial evolution',[(self.parameter,COLORS[3],values)],False
        return f'{self.parameter} vs fitness', [('Fitness',COLORS[0],[(float(p['hyperparameters'][self.parameter]),p['fitness'])
                for p in trials if self.parameter in p.get('hyperparameters',{}) and not p['failed']])],True

    def _plot(self,index,x,y,w,h,hovered=False,enlarged=False):
        self.create_rectangle(x+1,y+1,x+w-1,y+h-1,fill='white',outline='#6CA3FF' if hovered else BORDER,
                              width=3 if hovered else 1)
        title, lines, score = self._lines(index)
        self.create_text(x+12,y+15,text=title,anchor='w',fill='#19263B',font=('Segoe UI',11 if enlarged else 9,'bold'))
        plot_left=x+47
        plot_right=x+w-15
        plot_top=y+53
        plot_bottom=y+h-29
        if plot_right-plot_left <50 or plot_bottom-plot_top<25:
            return
        xx=[a for _,_,points in lines for a,_ in points]
        yy=[b for _,_,points in lines for _,b in points]
        if not xx:
            self.create_text(x+w/2,y+h/2,text='Waiting for completed trials',fill='#7D8DA2',font=('Segoe UI',10))
            return
        xmin,xmax=min(xx),max(xx)
        ymin,ymax=min(yy),max(yy)
        if score:
            ymin=min(0,ymin)
            ymax=max(1e-6,ymax)
        elif ymin==ymax:
            ymin=ymin-abs(ymin)*.1-.001
            ymax=ymax+abs(ymax)*.1+.001
        if xmin==xmax:
            xmin-=1
            xmax+=1
        def xp(v): return plot_left+(v-xmin)/(xmax-xmin)*(plot_right-plot_left)
        def yp(v): return plot_bottom-(v-ymin)/(ymax-ymin)*(plot_bottom-plot_top)
        for val in ticks(ymin,ymax,4):
            yy2=yp(val)
            self.create_line(plot_left,yy2,plot_right,yy2,fill='#E7EDF7',dash=(2,3))
            self.create_text(plot_left-5,yy2,text=f'{val:.3g}',anchor='e',font=('Segoe UI',8),fill='#7588A5')
        for tick in ticks(min(xx),max(xx),7,integers=index!=3):
            xp2=xp(tick)
            self.create_line(xp2,plot_bottom,xp2,plot_bottom+4,fill='#9FADC2')
            self.create_text(xp2,plot_bottom+13,text=f'{tick:.3g}' if index==3 else str(int(tick)),
                             fill='#6C7F9F',font=('Segoe UI',8))
        for label,col,points in lines:
            coords=[]
            for value_x,value_y in points:
                coords.extend((xp(value_x),yp(value_y)))
            if index!=3 and len(coords)>=4:
                self.create_line(*coords,fill=col,width=2,smooth=False)
            for j in range(0,len(coords),2):
                cx,cy=coords[j:j+2]
                self.create_oval(cx-2.8,cy-2.8,cx+2.8,cy+2.8,fill=col,outline='white',width=0.8)
        # Legend above plot, not on top of recorded values.
        xleg=plot_left
        for label,col,_ in lines:
            self.create_oval(xleg, y+34,xleg+7,y+41,fill=col,outline='')
            self.create_text(xleg+11,y+37,text=label,anchor='w',fill='#566E8E',font=('Segoe UI',8))
            xleg+=max(80,len(label)*6+35)
