#!/usr/local/bin/python
# -*- coding: utf-8 -*-
'''rcube
source code: Copyright (C) 2012, HatsuneMiku <999Hatsune@gmail.com>

for Vista
set PYGLET_SHADOW_WINDOW=0
'''

import sys, os
import traceback
import Queue
from collections import deque
import copy
import math
import random
import time
import ctypes
import pyglet
from pyglet import clock, font, image, media, window
from pyglet.gl import *
from OpenGL.GLUT import glutBitmapCharacter, GLUT_BITMAP_HELVETICA_10
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_12, GLUT_BITMAP_HELVETICA_18
from OpenGL.GLUT import GLUT_BITMAP_8_BY_13, GLUT_BITMAP_9_BY_15
from OpenGL.GLUT import GLUT_BITMAP_TIMES_ROMAN_10, GLUT_BITMAP_TIMES_ROMAN_24

FLAG_DEBUG = False
SHUFFLE_COUNT = 50
RESOURCE_PATH = 'resource'
TEXIMG_FACE, TEXIMG_HINT = 'f%d.bmp', '72dpi.bmp'
TEXIMG_CHAR = ['72dpi_ascii_reigasou_16x16.bmp']
FONT_FACE, FONT_FILE = u'みかちゃん-P'.encode('cp932'), 'mikaP.ttf'

INST_SHOW = u'i: to show instructions'
INSTRUCTIONS = u'''Instructions
i: these instructions on/off (auto off after %5.2f seconds)
z/y/x: rotate the cube clockwise (on/off toggle)
  (hold down shift to rotate counter-clockwise)
up/down/left/right: rotate the cube (shift to move slowly)
0: rotate the cube to home position
123/456/789: rotate single face or middle blocks clockwise
  (hold down shift to rotate counter-clockwise)
a: change animation speeds high/low
b: change alpha blend mode on/off
m: change texture mapping (show hint blocks)
p: printing internal state information on/off
page up/down: zoom in/out
F3: expand the cube
F4: shuffle the cube
F5: run the solver
l: load a suspended game
s: save this game suspended
custom function: 123456789
ESC: quit
'''

class MainWindow(window.Window):
  def __init__(self, *args, **kwargs):
    super(MainWindow, self).__init__(*args, **kwargs)
    self.keys = window.key.KeyStateHandler()
    self.push_handlers(self.keys)
    # self.set_exclusive_mouse()
    self.width, self.height, self.rat3d, self.ratex = 640, 480, 1.05, 0.5
    self.zoom, self.expand, self.mapping, self.blend = 0, 0, 0, 1
    self.fgc, self.bgc = (1.0, 1.0, 1.0, 0.9), (0.1, 0.1, 0.1, 0.1)
    self.loadfgc, self.loadbgc = (0.4, 0.2, 0.4, 0.3), (0.6, 0.3, 0.6, 0.9)
    self.instfgc, self.instbgc = (0.1, 0.1, 0.5, 0.9), (0.5, 0.9, 0.9, 0.8)
    self.instbkwidth, self.instbkheight = 480, 400
    bmplen = (self.instbkwidth / 8) * self.instbkheight
    self.instbkbmp = (ctypes.c_ubyte * bmplen)(*([255] * bmplen))
    self.ticktimer, self.tick, self.insttimer, self.inst = 0.5, 0.0, 30, 1
    self.printing, self.solver = 1, deque()
    self.stat = [None, 0, Queue.Queue(512)] # (key(1-9), direc), count, queue
    self.cmax, self.tanim = 18, [6, 3, 1, 3] # frames in rotate moving, speeds
    self.tcos, self.tsin = [1.0] * (self.cmax + 1), [0.0] * (self.cmax + 1)
    for i in xrange(1, self.cmax):
      t = i * math.pi / (2.0 * self.cmax) # 0 < t < pi/2
      self.tcos[i], self.tsin[i] = math.cos(t), math.sin(t)
    self.tcos[self.cmax], self.tsin[self.cmax] = 0.0, 1.0 # pi/2 regulation
    self.InitRot()
    self.InitAxis()
    self.InitGL(self.width, self.height)
    self.textures = [None] * (len(self.ary_norm) * 2 + 1 + len(TEXIMG_CHAR))
    self.loading, self.dat = 0, [('', 0, 0)] * len(self.textures)
    font.add_file('%s/%s' % (RESOURCE_PATH, FONT_FILE))
    self.font = font.load(FONT_FACE, 20)
    self.fontcolor = (0.5, 0.8, 0.5, 0.9)
    self.fps_display = clock.ClockDisplay(font=self.font, color=self.fontcolor)
    self.fps_pos = (-60.0, 30.0, -60.0)
    clock.set_fps_limit(60)
    clock.schedule_interval(self.update, 1.0 / 60.0)
    # self.LoadTextures() # call after (in self.update)

  def InitRot(self):
    self.rotdelta = 1.5
    self.rotxyz = [-45.0, -45.0, 0.0]
    self.keyxyz = [0, 0, 0]

  def InitAxis(self):
    self.ary_norm = [
      (0, 0, 1), ( 0,  0, -1), # z+ Front, z- Back
      (0, 1, 0), ( 0, -1,  0), # y+ Top, y- Bottom
      (1, 0, 0), (-1,  0,  0)] # x+ Right, x- Left
    self.ary_texc = [self.InitCell((0.00, 0.00), (0.30, 0.30), (0.35, 0.35),
      0, n) for n in xrange(6)]
    # self.ary_vtex = [self.InitCell((-0.9, -0.9), (0.54, 0.54), (0.63, 0.63),
    #   1, n) for n in xrange(6)]
    self.ary_vtex = [self.InitCell((-1.0, -1.0), (0.60, 0.60), (0.70, 0.70),
      1, n) for n in xrange(6)]
    self.ary_work = copy.deepcopy(self.ary_vtex)
    self.ary_pos = [[(n, c) for c in xrange(9)] for n in xrange(6)]
    self.ary_move = [
      (None, 0, '0000', '0000', '0000', '0000'), # 0: dummy
      (   0, 0, '4678', '2852', '5012', '3630'), # 1
      (None, 0, '4345', '2741', '5345', '3741'), # 2
      (   1, 0, '4012', '2630', '5678', '3852'), # 3
      (   2, 2, '0678', '4852', '1012', '5630'), # 4
      (None, 2, '0345', '4741', '1345', '5741'), # 5
      (   3, 2, '0012', '4630', '1678', '5852'), # 6
      (   4, 4, '2678', '0852', '3012', '1630'), # 7
      (None, 4, '2345', '0741', '3345', '1741'), # 8
      (   5, 4, '2012', '0630', '3678', '1852')] # 9

  def InitCell(self, offset, (w, h), span, mode, norm):
    if mode == 0: s, t = (-1, 1) if norm & 1 else (1, 1) # reverse s:LR (t:TB)
    else: s, t = (-1, -1) if norm & 1 else (1, 1) # reverse LR and TB
    a = [] # mode == 0: (lb,rb,rt,lt) x 9, mode == 1: [[n,c],[lb,rb,rt,lt]] x 9
    for r in xrange(3):
      for c in xrange(3):
        l0, b0 = s * (c * span[0] + offset[0]), t * (r * span[1] + offset[1])
        l1, b1 = l0 + s * w, b0 + t * h
        u = ((l0, b0), (l1, b0), (l1, b1), (l0, b1))
        if mode == 0: a.append(u)
        else: a.append([[norm, r * 3 + c],
          [self.Make3Dfrom2D(norm, u[i]) for i in xrange(4)]])
    return a

  def Make3Dfrom2D(self, axis, u):
    n = self.ary_norm[axis]
    p, q = u
    if n[2]: return [p, q, n[2] * self.rat3d] # z-xy
    elif n[1]: return [q, n[1] * self.rat3d, p] # y-zx
    else: return [n[0] * self.rat3d, p, q] # n[0] # x-yz

  def RotVtexReal(self, c, s, axis, u):
    n = self.ary_norm[axis]
    x, y, z = u
    if n[2]: return [x * c - y * s, x * s + y * c, z] # z-xy
    elif n[1]: return [z * s + x * c, y, z * c - x * s] # y-zx
    else: return [x, y * c - z * s, y * s + z * c] # n[0] # x-yz

  def RotVtex(self, direc, n_angle, axis, u):
    c, s = self.tcos[n_angle], direc * self.tsin[n_angle]
    return self.RotVtexReal(c, s, axis, u)

  def ReplaceVtexSub(self, direc, n_angle, axis, norm, *cells):
    for c in cells:
      pn, pc = self.ary_pos[norm][c]
      v, w = self.ary_vtex[pn][pc], self.ary_work[pn][pc]
      for i in xrange(4): v[1][i] = self.RotVtex(direc, n_angle, axis, w[1][i])

  def ReplaceVtex(self):
    if self.stat[0] is None:
      if self.stat[2].empty(): return
      self.stat[0] = self.stat[2].get()
    if not self.stat[1]: self.ary_work = copy.deepcopy(self.ary_vtex)
    self.stat[1] += self.tanim[0]
    if self.stat[1] >= self.cmax: self.stat[1] = self.cmax
    m, d, a = self.ary_move[self.stat[0][0]], self.stat[0][1], self.stat[1]
    if m[0] is not None: self.ReplaceVtexSub(d, a, m[0], m[0], *xrange(9))
    for i in xrange(2, 6): self.ReplaceVtexSub(d, a, m[1], *map(int, m[i]))
    if self.stat[1] >= self.cmax:
      q = copy.deepcopy(self.ary_pos)
      if m[0] is not None:
        n, o = m[0], ['630741852', '258147036']
        p = map(int, o[0 if self.stat[0][1] > 0 else 1]) # n & 1
        for i in xrange(9):
          pn, pc = self.ary_pos[n][i] = q[n][p[i]]
          self.ary_vtex[pn][pc][0] = [n, i]
      for i in xrange(2, 6):
        # self.ReplacePos(m[])
        if self.stat[0][1] > 0: # 5 - (i - 2), 5 if i == 5 else 4 - (i - 2)
          k, l = map(int, m[7 - i]), map(int, m[5 if i == 5 else (6 - i)])
        else:
          k, l = map(int, m[i]), map(int, m[2 if i == 5 else (i + 1)])
        for j in xrange(3):
          pn, pc = self.ary_pos[k[0]][k[1 + j]] = q[l[0]][l[1 + j]]
          self.ary_vtex[pn][pc][0] = [k[0], k[1 + j]]
      self.stat[0:2] = [None, 0]

  def InitGL(self, width, height):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL) # GL_MODULATE

  def LoadTextures(self):
    '''texture 0-5: faces, 6-11: hints, 12: blending screen, 13: char'''
    i = self.loading
    if i < len(self.textures): # for i in xrange(len(self.textures)):
      if i < len(self.ary_norm): imgfile = TEXIMG_FACE % i # bmp24 256x256
      elif i <= len(self.ary_norm) * 2: imgfile = TEXIMG_HINT
      else: imgfile = TEXIMG_CHAR[i - len(self.ary_norm) * 2 - 1]
      img = image.load('%s/%s' % (RESOURCE_PATH, imgfile))
      self.textures[i] = img.get_texture()
      ix, iy = img.width, img.height
      rawimage = img.get_image_data()
      formatstr = 'RGBA'
      pitch = rawimage.width * len(formatstr)
      dat = rawimage.get_data(formatstr, pitch)
      self.dat[i] = (dat, ix, iy)
      if i > len(self.ary_norm): # skip face(0-5) and hint(6:white)
        j = i - len(self.ary_norm)
        d = []
        it = iter(dat)
        while it.__length_hint__():
          r, g, b, a = [ord(it.next()) for k in xrange(4)]
          if i < len(self.ary_norm) * 2 and r >= 128 and g >= 128 and b >= 128:
            r, g, b = r if j & 1 else 0, g if j & 2 else 0, b if j & 4 else 0
          elif i == len(self.ary_norm) * 2:
            r, g, b, a = [int(self.instbgc[k] * 255) for k in xrange(4)]
          else:
            r, g, b, a = r, g, b, 255 * (r + g + b) / (255 * 3)
          d.append('%c%c%c%c' % (r, g, b, a))
        dat = ''.join(d)
      glEnable(self.textures[i].target)
      glBindTexture(self.textures[i].target, self.textures[i].id)
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
      glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, dat)
      self.loading = i + 1

  def ModifyTexture(self, n):
    if self.loading <= n: return
    mode = 0
    d, ix, iy = self.dat[n]
    ox, oy, w, h = ix / 2 - 96, iy / 2 - 54, 192, 48
    col = [int(f * 255) for f in (0.1, 0.9, 0.8, 0.1)]
    if mode == 1: # 30 fps (too late)
      buf = list(d)
      for y in xrange(h):
        for x in xrange(w):
          p = ((oy + y) * ix + (ox + x)) * len('RGBA')
          buf[p:p+len('RGBA')] = map(chr, col)
      d = ''.join(buf)
    # glEnable(self.textures[n].target)
    glBindTexture(self.textures[n].target, self.textures[n].id)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0,
      GL_RGBA, GL_UNSIGNED_BYTE, d)
    if mode == 0: # 40-60 fps (fast) but 22-34 fps (6 chars)
      buf = map(chr, col) * (w * h)
      # self.DrawBlendStringOnBuffer(buf, (w, h), (60, 30), 'NiPpOn')
      s = ('NiPpOn', (60,30), ((0,1), (10,0), (20,1), (32,0), (46,1), (58,0)))
      for i, c in enumerate(s[0]): self.DrawBlendCharOnBuffer(buf, (w, h),
        (s[1][0] + s[2][i][0], s[1][1] + s[2][i][1]), c)
      t = time.time()
      fmt = '%H:%M:%S' if t - int(t) < .5 else '%H %M %S'
      s = time.strftime(fmt, time.localtime(t))
      self.DrawBlendStringOnBuffer(buf, (w, h), (4, 15), s, (8, 8))
      glTexSubImage2D(GL_TEXTURE_2D, 0, ox, oy, w, h,
        GL_RGBA, GL_UNSIGNED_BYTE, ''.join(buf))

  def DrawTextureQuad(self, ary_t2f, ary_v3f, num=0, sfact=None, dfact=None):
    if sfact is None: sfact = GL_SRC_ALPHA
    if dfact is None: dfact = GL_ONE_MINUS_SRC_ALPHA
    if ary_t2f is None: u = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.1), (0.0, 1.1))
    else: u = ary_t2f
    if self.blend:
      # glAlphaFunc(GL_GEQUAL, 0.4)
      # glEnable(GL_ALPHA_TEST)
      glColor4f(*self.fgc)
      # glBlendFunc(GL_DST_ALPHA, GL_ONE)
      # glBlendFunc(GL_SRC_ALPHA, GL_ONE)
      # glBlendFunc(GL_ONE, GL_DST_ALPHA)
      # glBlendFunc(GL_ONE, GL_SRC_ALPHA)
      # glBlendFunc(GL_ONE, GL_ONE_MINUS_DST_ALPHA)
      # glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
      # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
      # glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
      # glBlendFunc(GL_ONE_MINUS_SRC_COLOR, GL_ONE_MINUS_DST_COLOR)
      # glBlendFunc(GL_SRC_COLOR, GL_DST_COLOR)
      # glBlendFunc(GL_SRC_ALPHA, GL_DST_COLOR)
      glBlendFunc(sfact, dfact)
      glEnable(GL_BLEND)
    else:
      # glDisable(GL_ALPHA_TEST)
      # glColor4f(1.0, 1.0, 1.0, 1.0)
      glDisable(GL_BLEND)
    if self.loading <= num: return
    glEnable(GL_TEXTURE_2D)
    glBindTexture(self.textures[num].target, self.textures[num].id)
    glBegin(GL_QUADS)
    for i, p in enumerate(u): glTexCoord2f(*p); glVertex3f(*ary_v3f[i])
    glEnd()
    glDisable(GL_TEXTURE_2D)

  def DrawGLPlane(self, n, sfact=GL_SRC_ALPHA, dfact=GL_SRC_COLOR):
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    for c in xrange(9):
      t, v = self.ary_texc[n][c], self.ary_vtex[n][c]
      pn, pc = v[0]
      u = self.ary_norm[pn]
      if self.expand and self.stat[0]:
        m, d, a = self.ary_move[self.stat[0][0]], self.stat[0][1], self.stat[1]
        m2nc = [[]] * len(self.ary_norm)
        for i in xrange(2, 6): m2nc[int(m[i][0])] = map(int, m[i][1:])
        if pc in m2nc[pn]: u = self.RotVtex(d, a, m[1], u)
      w = [[(v[1][i][j] + u[j] * self.ratex * self.expand) \
        for j in xrange(3)] for i in xrange(4)]
      m = n + self.mapping * len(self.ary_norm)
      self.DrawTextureQuad(t, w, m, sfact, dfact)

  def DrawBlendScreen(self, w, h, sfact=GL_SRC_ALPHA, dfact=GL_SRC_COLOR):
    u = [0.0, 0.0, 1.8]
    m, r = 1.2 * [1.9370, 1.0000, 0.0625][1 + self.zoom], float(w) / float(h)
    l0, b0, l1, b1, z = -m * r, -m, m * r, m, u[2]
    v = [[l0, b0, z], [l1, b0, z], [l1, b1, z], [l0, b1, z]]
    for i in xrange(len(self.ary_norm) / 2):
      t = -self.rotxyz[i] * math.pi / 180.0
      c, s = math.cos(t), math.sin(t)
      axis = len(self.ary_norm) - (i + 1) * 2
      u = self.RotVtexReal(c, s, axis, u)
      v = [self.RotVtexReal(c, s, axis, v[j]) for j in xrange(4)]
    if True: # 33-34 fps
      self.DrawTextureQuad(None, v, len(self.ary_norm) * 2, sfact, dfact)
    else: # 33-34 fps
      glColor4f(*self.instbgc)
      glBegin(GL_POLYGON)
      for j in xrange(4): glVertex3f(*v[j])
      glEnd()
    return u

  def DrawBlendCharOnVertex3f(self, ary_v3f, c='@', num=0, w=16, h=16):
    r = self.GetCharBmpPosOnTexture(c, num, w, h)
    if r is None: return
    m, d, ix, iy, l0, b0, l1, b1 = r
    l0, b0 = float(l0) / ix, float(b0) / iy
    l1, b1 = float(l1) / ix, float(b1) / iy
    u = ((l0, b0), (l1, b0), (l1, b1), (l0, b1))
    self.DrawTextureQuad(u, ary_v3f, m, GL_SRC_COLOR, GL_SRC_ALPHA)

  def DrawBlendCharOnBuffer(self, buf, (bw, bh), pos=(0, 0),
    c='@', num=0, w=16, h=16):
    _ = lambda s, d, a: max(0, min(255, (s + d))) # (s * (1 - a) + d * a) / 255
    r = self.GetCharBmpPosOnTexture(c, num, w, h)
    if r is None: return
    m, d, ix, iy, l0, b0, l1, b1 = r
    for y in xrange(h):
      for x in xrange(w):
        q = ((b0 + y) * ix + (l0 + x)) * len('RGBA')
        p = ((pos[1] + y) * bw + (pos[0] + x)) * len('RGBA')
        if False: # 53-54 fps
          srgba = [ord(c) for c in d[q:q+len('RGBA')]]
          drgba = [ord(c) for c in buf[p:p+len('RGBA')]]
          rgba = [_(srgba[i], drgba[i], srgba[3]) for i in xrange(len('RGBA'))]
          buf[p:p+len('RGBA')] = map(chr, rgba)
        # 57-60 fps
        sr, sg, sb, sa = [ord(c) for c in d[q:q+len('RGBA')]]
        dr, dg, db, da = [ord(c) for c in buf[p:p+len('RGBA')]]
        r, g, b, a = _(sr, dr, sa), _(sg, dg, sa), _(sb, db, sa), _(sa, da, sa)
        buf[p:p+len('RGBA')] = map(chr, (r, g, b, a))

  def GetCharBmpPosOnTexture(self, c='@', num=0, w=16, h=16):
    m = len(self.ary_norm) * 2 + 1 + num
    if self.loading <= m: return None
    d, ix, iy = self.dat[m]
    fchr = 16 # sqrt(256:characters)
    l0, b0 = (ord(c) % fchr) * w, (fchr - 1 - (ord(c) / fchr)) * h
    l1, b1 = l0 + w, b0 + h
    return m, d, ix, iy, l0, b0, l1, b1

  def DrawBlendStringOnVertex3f(self, ary_v3f,
    s='', spc=(0, 0), num=0, w=16, h=16):
    for i, c in enumerate(s):
      self.DrawBlendCharOnVertex3f(ary_v3f, c, num, w, h)

  def DrawBlendStringOnBuffer(self, buf, (bw, bh), pos=(0, 0),
    s='', spc=(0, 0), num=0, w=16, h=16):
    for i, c in enumerate(s): self.DrawBlendCharOnBuffer(buf, (bw, bh),
      (pos[0] + i * (w + spc[0]), pos[1]), c, num, w, h)

  def DrawString(self, s='', f=None):
    if f is None: f = GLUT_BITMAP_HELVETICA_12
    for c in s: glutBitmapCharacter(f, ord(c))

  def DrawString3f(self, pos=(0.0, 0.0, 0.0), s='', f=None):
    glRasterPos3f(*pos); self.DrawString(s, f)

  def DrawString2f(self, pos=(0.0, 0.0), s='', f=None):
    glRasterPos2f(*pos); self.DrawString(s, f)

  def DrawStringToScreen(self, pos=(0.0, 0.0), s='', f=None):
    glWindowPos2f(*pos); self.DrawString(s, f)

  def DrawMultilineToScreen(self, h, pos=(0.0, 0.0), s='', f=None):
    lines = s.split('\n')
    m = len(lines) - 1
    for i, l in enumerate(lines):
      self.DrawStringToScreen((pos[0], pos[1] + h * (m - i)), l, f)

  def DrawLoading(self):
    glDisable(GL_TEXTURE_2D)
    f24 = GLUT_BITMAP_TIMES_ROMAN_24
    w, h = 320, 80 # must be small than self.instbkwidth, self.instbkheight
    glColor4f(*self.loadbgc)
    # glRasterPos2f(0.0, 0.0) # z = 0.0 (pushed away)
    glRasterPos3f(0.9, -1.2727922, 0.9) # (-45.0, -45.0, 0.0)[0.0, 0.0, 1.8]
    glBitmap(w, h, w / 2, h / 2, 0.0, 0.0, self.instbkbmp) # use bmp partial
    glColor4f(*self.loadfgc)
    self.DrawStringToScreen((200, 230), 'Loading texture %d of %d...' % (
      self.loading + 1, len(self.textures)), f24)
    glEnable(GL_TEXTURE_2D)

  def DrawInstructions(self):
    if not self.inst:
      self.DrawStringToScreen((10.0, 10.0), INST_SHOW)
      return
    if self.loading <= len(self.ary_norm) * 2: return
    glDisable(GL_TEXTURE_2D)
    f10, f18 = GLUT_BITMAP_HELVETICA_10, GLUT_BITMAP_HELVETICA_18
    w, h = self.instbkwidth, self.instbkheight
    mode = 0
    if mode > 0: glColor4f(*self.instbgc)
    if mode == 0: # 35-40 fps (ok and display u)
      u = self.DrawBlendScreen(w, h)
    elif mode == 1: # 15-25 fps (bug: vibration) (too late)
      u = [0.0, 0.0, 0.0] # dummy
      glRasterPos3f(*u) # u must be got from that in self.DrawBlendScreen
    elif mode == 2: # 20 fps (30 fps when origin (0.0, 0.0) because half area)
      glRasterPos2f(0.0, 0.0) # z = 0.0
    else: # mode == 3 # 15-20 fps (bug: overlapped strings are not shown)
      glWindowPos2f(self.width / 2, self.height / 2) # z = 0.0
    if mode > 0: glBitmap(w, h, w / 2, h / 2, 0.0, 0.0, self.instbkbmp)
    glColor4f(*self.instfgc)
    lb = ((self.width - w) / 2, (self.height - h) / 2)
    self.DrawStringToScreen(lb,
      'rot (%7.2f, %7.2f, %7.2f)' % tuple(self.rotxyz), f10)
    if mode <= 1: self.DrawStringToScreen((lb[0] + w / 2, lb[1]),
      'screen (%12.9f, %12.9f, %12.9f)' % tuple(u), f10)
    remain = self.insttimer - self.inst
    self.DrawMultilineToScreen(19, lb, INSTRUCTIONS % remain, f18)
    glEnable(GL_TEXTURE_2D)

  def DrawInternalState(self):
    if not self.printing: return
    glDisable(GL_TEXTURE_2D)
    f10, f24 = GLUT_BITMAP_TIMES_ROMAN_10, GLUT_BITMAP_TIMES_ROMAN_24
    self.DrawClockFPS()
    glColor4f(*self.fgc)
    self.DrawString3f(self.fps_pos, str(self.fps_pos))
    self.DrawString3f((-2.0, -2.0, -2.0), '(-2.0, -2.0, -2.0)', f10)
    self.DrawString3f((2.0, 2.0, 2.0), '(2.0, 2.0, 2.0)', f10)
    self.DrawString2f((-1.0, -1.0), '(-1.0, -1.0)', f24)
    self.DrawString2f((1.0, 1.0), '(1.0, 1.0)', f24)
    glEnable(GL_TEXTURE_2D)

  def DrawClockFPS(self):
    fpslbl = self.fps_display.label
    fpslbl.x, fpslbl.y, fpslbl.z = self.fps_pos
    # print fpslbl.width, fpslbl.height # 122, 56
    self.fps_display.draw()

  def DrawGLAxis(self, n):
    glDisable(GL_TEXTURE_2D)
    glBegin(GL_LINES)
    glColor3f(*(reversed(self.ary_norm[n]) if n else (1, 1, 1)))
    glVertex3f(0.0, 0.0, 0.0); glVertex3f(*[a * 5.0 for a in self.ary_norm[n]])
    glEnd()
    if n == 0:
      self.DrawString2f((0.0, 0.0), '%s fps' % self.fps_display.label.text)
    glEnable(GL_TEXTURE_2D)

  def DrawGLScene(self):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5.0 + 3.0 * self.zoom)
    glRotatef(self.rotxyz[0], 1.0, 0.0, 0.0)
    glRotatef(self.rotxyz[1], 0.0, 1.0, 0.0)
    glRotatef(self.rotxyz[2], 0.0, 0.0, 1.0)
    self.DrawInternalState()
    for n in xrange(3): self.DrawGLAxis(n * 2)
    # Must draw face planes right sequence for alpha blending. (pn, pc)
    for n in xrange(6): self.DrawGLPlane(len(self.ary_norm) - 1 - n)
    self.DrawInstructions()
    if self.loading < len(self.textures): self.DrawLoading()
    for i in xrange(3):
      self.rotxyz[i] += self.keyxyz[i] * self.rotdelta
      if self.rotxyz[i] <= -180.0: self.rotxyz[i] = 180.0
      elif self.rotxyz[i] >= 180.0: self.rotxyz[i] = -180.0

  def update(self, dt):
    if FLAG_DEBUG:
      s = traceback.extract_stack()[-1]
      print s[0], s[1], self.__class__.__name__, s[2], dt
      print sys._getframe(0).f_code # .co_name
      print sys._getframe(1).f_code # .co_name
    if self.loading < len(self.textures): self.LoadTextures()
    if self.inst > 0: self.inst += dt
    if self.inst > self.insttimer: self.inst = 0
    self.tick += dt
    if self.tick >= self.ticktimer: self.tick = 0.0
    if not self.mapping and self.tick <= 0.0: self.ModifyTexture(0)
    self.ReplaceVtex()

  def on_resize(self, width, height):
    if width == 0: width = 1
    if height == 0: height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(width)/float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    # super(MainWindow, self).on_resize(width, height)

  def on_draw(self):
    dt = clock.tick()
    # print dt
    self.DrawGLScene()

  def on_key_press(self, symbol, modifiers):
    direc = -1 if self.keys[window.key.LSHIFT] else 1
    if self.chcmp(symbol, 'Ii'): self.inst = 0 if self.inst > 0 else 1
    elif self.chcmp(symbol, 'XYZxyz'):
      c = symbol - (ord('X') if symbol <= ord('Z') else ord('x'))
      self.keyxyz[c] = 0 if self.keyxyz[c] else -direc
    elif symbol == window.key.UP: self.keyxyz[0] = 1.5 + direc
    elif symbol == window.key.DOWN: self.keyxyz[0] = -(1.5 + direc)
    elif symbol == window.key.LEFT: self.keyxyz[1] = 1.5 + direc
    elif symbol == window.key.RIGHT: self.keyxyz[1] = -(1.5 + direc)
    elif symbol == ord('0'): self.InitRot()
    elif self.chcmp(symbol, '123456789'): self.move(symbol - ord('0'), -direc)
    elif self.chcmp(symbol, 'Aa'): self.tanim = self.tanim[1:] + self.tanim[:1]
    elif self.chcmp(symbol, 'Bb'): self.blend = 1 - self.blend
    elif self.chcmp(symbol, 'Mm'): self.mapping = 1 - self.mapping
    elif self.chcmp(symbol, 'Pp'): self.printing = 1 - self.printing
    elif symbol == window.key.PAGEUP: self.zoom = 0 if self.zoom < 0 else 1
    elif symbol == window.key.PAGEDOWN: self.zoom = 0 if self.zoom > 0 else -1
    elif symbol == window.key.F3: self.expand = 1 - self.expand
    elif symbol == window.key.F4: self.shuffle()
    elif symbol == window.key.F5: self.solve()
    elif symbol == window.key.ESCAPE: self.dispatch_event('on_close')
    else: pass

  def chcmp(self, symbol, chs):
    for ch in chs:
      if symbol == ord(ch): return True
    return False

  def move(self, k, d):
    if self.stat[2].full(): print u'--- key(1-9) queue is full ---'
    else:
      self.stat[2].put((k, d))
      try:
        self.solver.append((k, -d))
      except:
        print u'\a--- solver queue is full ---'

  def shuffle(self):
    for i in xrange(SHUFFLE_COUNT):
      self.move(random.randint(1, 9), -1 if random.randint(0, 1) else 1)

  def solve(self):
    while len(self.solver):
      if self.stat[2].full():
        print u'--- to be continued (please retry to push F5 later) ---'
        break
      else: self.stat[2].put(self.solver.pop())

if __name__ == '__main__':
  # pyglet.resource.path = [RESOURCE_PATH]
  # pyglet.resource.reindex()
  w = MainWindow(caption='rcube', fullscreen=False, resizable=True)
  pyglet.app.run()
