#!/usr/bin/env python3
"""
Rosa ASCII animada.  Python + NumPy + pygame-ce (o terminal).

Proceso
  Forma     el tallo, las hojas, los sépalos y un capullo cerrado se construyen en 3D.
  Crecimiento  un punto de luz sube dibujando el tallo y después se llena el capullo.
  Floración una sola superficie espiral se suelta del capullo hasta ser una rosa abierta,
            primero los pétalos de afuera.
  Movimiento balanceo suave con los pétalos rezagados, gotas de rocío que caen y salpican.
  ASCII     la escena se convierte en una cuadrícula de caracteres sombreados por brillo.

Técnicas: geometría 3D y rotación, superficies paramétricas en espiral, proyección en
perspectiva con orden de profundidad (z-buffer), luz difusa y especular, partículas con
gravedad, detección de contornos y sombreado con caracteres.

Uso
  python rosa_ascii.py                        ventana pygame (o terminal si no hay pantalla)
  python rosa_ascii.py --modo terminal        animación en la terminal (ANSI)
  python rosa_ascii.py --png rosa.png --t 16  un fotograma como imagen
  Teclas (pygame): ESC salir · ESPACIO pausa · R reiniciar · ←/→ ±1 s · +/- velocidad
"""

import argparse
import math
import os
import signal
import sys
import time

import numpy as np

# --------------------------------------------------------------------------- línea de tiempo
T_CRECE = 5.0            # el punto de luz termina de subir por el tallo
T_YEMA_INI = 4.6         # empieza a llenarse el capullo
T_YEMA_FIN = 7.4
T_FLORECE_INI = 7.5
T_FLORECE_FIN = 15.0     # floración 100 %
T_CICLO = 30.0           # a los 30 s se disuelve y vuelve a empezar
T_DISUELVE = 1.6
CICLO_BALANCEO = 16.0    # segundos por vuelta de la fase del balanceo (360°)

# --------------------------------------------------------------------------- aspecto
# De oscuro a claro (la densidad del carácter marca el brillo).
RAMPA = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
RAMPA_ARR = np.frombuffer(RAMPA.encode("ascii"), dtype=np.uint8)

SUELO_Y = -3.35
TALLO_ARRIBA_Y = -0.10
ALTO_PLANTA = 3.8
OBJETIVO = np.array([0.0, -1.38, 0.0], dtype=np.float32)
LUZ_DIR = np.array([0.45, 0.72, 0.52], dtype=np.float32)
LUZ_DIR /= np.linalg.norm(LUZ_DIR)
MEDIO = LUZ_DIR + np.array([0, 0, 1], dtype=np.float32)
MEDIO /= np.linalg.norm(MEDIO)

PETALO_INT = np.array([205, 24, 46], dtype=np.float32)
PETALO_EXT = np.array([255, 52, 64], dtype=np.float32)
VERDE_OSC = np.array([22, 150, 66], dtype=np.float32)
VERDE_CLA = np.array([70, 250, 120], dtype=np.float32)
ROCIO = np.array([175, 228, 255], dtype=np.float32)
LUZ = np.array([225, 255, 235], dtype=np.float32)


def suave(a, b, x):
    t = np.clip((np.asarray(x, dtype=np.float32) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def mezcla(a, b, t):
    t = np.asarray(t, dtype=np.float32)[..., None]
    return a * (1.0 - t) + b * t


def normalizar(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def normales_malla(pos):
    """Normales de una malla (n_u, n_v, 3) a partir de sus derivadas."""
    du = np.gradient(pos, axis=0)
    dv = np.gradient(pos, axis=1)
    return normalizar(np.cross(du, dv))


# --------------------------------------------------------------------------- nube de puntos
class Nube:
    """Puntos con normal, color base, brillo especular, oclusión y luz propia."""

    def __init__(self):
        self.p, self.n, self.rgb, self.esp, self.bri, self.ao, self.ext = ([] for _ in range(7))
        self.bor, self.gli = [], []

    def agregar(self, p, n, rgb, esp, bri, ao=1.0, ext=0.0, borde=1.0, glifo=0):
        p = np.asarray(p, dtype=np.float32).reshape(-1, 3)
        k = len(p)
        if k == 0:
            return
        self.p.append(p)
        self.n.append(np.asarray(n, dtype=np.float32).reshape(-1, 3))
        self.rgb.append(np.broadcast_to(np.asarray(rgb, dtype=np.float32), (k, 3)).reshape(-1, 3))
        self.esp.append(np.broadcast_to(np.float32(esp), (k,)).astype(np.float32))
        self.bri.append(np.broadcast_to(np.float32(bri), (k,)).astype(np.float32))
        self.ao.append(np.broadcast_to(np.asarray(ao, dtype=np.float32), (k,)).astype(np.float32))
        self.ext.append(np.broadcast_to(np.asarray(ext, dtype=np.float32), (k,)).astype(np.float32))
        self.bor.append(np.broadcast_to(np.float32(borde), (k,)).astype(np.float32))
        self.gli.append(np.broadcast_to(np.uint8(glifo), (k,)).astype(np.uint8))

    def unir(self):
        if not self.p:
            return None
        return [np.concatenate(x) for x in (self.p, self.n, self.rgb, self.esp, self.bri, self.ao, self.ext, self.bor, self.gli)]


# --------------------------------------------------------------------------- geometría de la planta
class Planta:
    """Genera la nube de puntos de la rosa para un instante t."""

    def __init__(self, densidad=1.0, semilla=7):
        rng = np.random.default_rng(semilla)
        self.rng = rng
        d = max(0.35, densidad)

        # --- tallo
        self.ns = int(230 * d)
        self.nt = 16
        self.ts = np.linspace(0, 1, self.ns, dtype=np.float32)
        self.th = np.linspace(0, 2 * np.pi, self.nt, endpoint=False, dtype=np.float32)

        # --- rosa (una sola espiral)
        self.vueltas = 4.6
        self.phi_max = self.vueltas * 2 * np.pi
        self.n_phi = int(1180 * d)
        self.n_v = int(40 * d)
        self.phi = np.linspace(0.35, self.phi_max, self.n_phi, dtype=np.float32)[:, None]
        self.vv = np.linspace(0, 1, self.n_v, dtype=np.float32)[None, :]
        self.petalos = 5.35                       # pétalos por vuelta (no entero: se escalonan)
        self.tam_flor = 0.68
        self._cache_rosa = None
        self.fase_petalo = rng.uniform(0, 6.28, size=1).astype(np.float32)

        # --- hojas: (posición en el tallo, azimut, tamaño, momento en que el punto de luz pasa)
        self.hojas = [
            dict(tp=0.30, psi=np.pi - 0.18, tam=1.18, pasa=0.30),
            dict(tp=0.55, psi=0.20, tam=1.05, pasa=0.55),
        ]
        self.nm, self.nw = int(64 * d), int(24 * d)

        # --- sépalos
        self.nsep = 5
        self.ns_l, self.ns_w = int(34 * d), int(9 * d)

        # --- gotas de rocío (ciclo propio de cada una)
        self.gotas = self._crear_gotas(rng)
        self._t_actual = 0.0
        self._puntas_hoja = None
        esfera = np.arange(52, dtype=np.float32) + 0.5
        cz = 1 - 2 * esfera / 52
        ang = np.pi * (1 + 5 ** 0.5) * esfera
        r = np.sqrt(1 - cz * cz)
        self.esfera = np.stack([r * np.cos(ang), cz, r * np.sin(ang)], axis=1).astype(np.float32)

        # umbral aleatorio estable para disolver al final del ciclo
        self.umbral = rng.random(400000).astype(np.float32)

        # anclajes de las gotas sobre la flor abierta (bloom = 1)
        self._anclas_flor = None

    # ---------------------------------------------------------------- balanceo
    @staticmethod
    def fase_balanceo(t):
        return 2 * np.pi * (t % CICLO_BALANCEO) / CICLO_BALANCEO

    def balanceo(self, p, t, rezago=0.0):
        """Doblado suave del tallo hacia un lado y otro; los pétalos van rezagados."""
        fase = self.fase_balanceo(t) - rezago
        h = np.clip((p[:, 1] - SUELO_Y) / (ALTO_PLANTA + 0.1), 0, 1.3) ** 1.7
        out = p.copy()
        out[:, 0] += 0.17 * np.sin(fase) * h
        out[:, 2] += 0.08 * np.sin(fase + 1.1) * h
        return out

    # ---------------------------------------------------------------- tallo
    @staticmethod
    def centro_tallo(tp):
        tp = np.asarray(tp, dtype=np.float32)
        x = 0.13 * np.sin(2.4 * tp + 0.4) - 0.05 + 0.04 * tp
        z = 0.07 * np.sin(3.1 * tp)
        y = SUELO_Y + (TALLO_ARRIBA_Y - SUELO_Y) * tp
        return np.stack([x, y, z], axis=-1)

    def tallo(self, nube, avance):
        n_vis = int(self.ns * avance)
        if n_vis < 3:
            return
        ts = self.ts[:n_vis]
        c = self.centro_tallo(ts)
        tang = normalizar(np.gradient(c, axis=0))
        ref = np.array([0, 0, 1], dtype=np.float32)
        nn = normalizar(np.cross(tang, ref))
        bb = np.cross(tang, nn)
        radio = 0.098 - 0.026 * ts + 0.035 * np.clip(1 - ts / 0.06, 0, 1)
        co, si = np.cos(self.th)[None, :], np.sin(self.th)[None, :]
        dirs = co[..., None] * nn[:, None, :] + si[..., None] * bb[:, None, :]
        pos = c[:, None, :] + radio[:, None, None] * dirs
        # el pie se difumina en el suelo
        vis = suave(0.0, 0.10, ts)[:, None] * np.ones((1, self.nt), dtype=np.float32)
        col = mezcla(VERDE_OSC, VERDE_CLA, 0.78 + 0.2 * np.sin(self.th)[None, :] + 0 * ts[:, None])
        nube.agregar(pos.reshape(-1, 3), dirs.reshape(-1, 3), col.reshape(-1, 3), 0.30, 18,
                     ao=(0.55 + 0.45 * vis).reshape(-1), ext=(0.22 * vis).reshape(-1))

    # ---------------------------------------------------------------- hojas
    def _hoja_malla(self, h, abre):
        escala = h["tam"] * (0.25 + 0.75 * abre)
        s = np.linspace(0, 1, self.nm, dtype=np.float32)
        w = np.linspace(-1, 1, self.nw, dtype=np.float32)
        elev = (1.35 - 0.50 * abre) - 1.75 * abre * s ** 1.3
        ds = 1.05 * escala / (self.nm - 1)
        sal = np.cumsum(np.cos(elev)) * ds
        alt = np.cumsum(np.sin(elev)) * ds
        ancho = 0.36 * escala * np.clip(np.sin(np.pi * s ** 0.8), 0, None) ** 0.85 * (1 + 0.09 * np.sin(24 * np.pi * s))
        ancho = ancho * (0.30 + 0.70 * abre)
        psi = h["psi"]
        dh = np.array([np.cos(psi), 0, np.sin(psi)], dtype=np.float32)
        pp = np.array([-np.sin(psi), 0, np.cos(psi)], dtype=np.float32)
        arriba = np.array([0, 1, 0], dtype=np.float32)
        ancla = self.centro_tallo(h["tp"]) + dh * 0.05
        pos = (ancla[None, None, :]
               + sal[:, None, None] * dh[None, None, :]
               + alt[:, None, None] * arriba[None, None, :]
               + (w[None, :] * ancho[:, None])[..., None] * pp[None, None, :]
               - (np.abs(w)[None, :] * ancho[:, None] * 0.42)[..., None] * arriba[None, None, :])
        pos[..., 1] += 0.025 * np.sin(11 * s)[:, None] * np.cos(3 * w)[None, :]
        return pos, s

    def hoja(self, nube, h, avance_tallo):
        crece = suave(0.0, 0.32, avance_tallo - h["pasa"])
        if crece <= 0.01:
            return
        abre = float(suave(0.0, 1.0, crece))
        pos, s = self._hoja_malla(h, abre)
        pos_b = self.balanceo(pos.reshape(-1, 3), self._t_actual).reshape(pos.shape)
        nor = normales_malla(pos_b)
        uno = np.ones((1, self.nw), dtype=np.float32)
        col = mezcla(VERDE_OSC, VERDE_CLA, (0.45 + 0.55 * s)[:, None] * uno)
        ao = 0.70 + 0.30 * suave(0.0, 0.30, s)[:, None] * uno
        nube.agregar(pos_b.reshape(-1, 3), nor.reshape(-1, 3), col.reshape(-1, 3), 0.25, 26, ao=ao.reshape(-1), ext=0.10)

    # ---------------------------------------------------------------- sépalos y receptáculo
    def sepalos(self, nube, crece, bloom):
        if crece <= 0.01:
            return
        s = np.linspace(0, 1, self.ns_l, dtype=np.float32)
        w = np.linspace(-1, 1, self.ns_w, dtype=np.float32)
        # pegados al capullo y luego se doblan hacia abajo
        elev = (1.32 - 2.65 * float(suave(0.10, 1.0, bloom))) - 0.40 * s
        largo = 0.52 * float(suave(0, 1, crece))
        ds = largo / (self.ns_l - 1)
        sal = np.cumsum(np.cos(elev)) * ds
        alt = np.cumsum(np.sin(elev)) * ds
        ancho = 0.075 * np.clip(np.sin(np.pi * s ** 0.7), 0, None) ** 0.8 + 0.01 * (1 - s)
        for k in range(self.nsep):
            psi = 2 * np.pi * k / self.nsep + 0.4
            dh = np.array([np.cos(psi), 0, np.sin(psi)], dtype=np.float32)
            pp = np.array([-np.sin(psi), 0, np.cos(psi)], dtype=np.float32)
            arriba = np.array([0, 1, 0], dtype=np.float32)
            ancla = np.array([0, TALLO_ARRIBA_Y + 0.06, 0], dtype=np.float32) + dh * 0.10
            pos = (ancla[None, None, :]
                   + sal[:, None, None] * dh[None, None, :]
                   + alt[:, None, None] * arriba[None, None, :]
                   + (w[None, :] * ancho[:, None])[..., None] * pp[None, None, :]
                   - (np.abs(w)[None, :] * ancho[:, None] * 0.5)[..., None] * arriba[None, None, :])
            nor = normales_malla(pos)
            col = mezcla(VERDE_OSC, VERDE_CLA, 0.2 + 0.3 * s)[:, None, :] * np.ones((1, self.ns_w, 1), dtype=np.float32)
            nube.agregar(pos.reshape(-1, 3), nor.reshape(-1, 3), col.reshape(-1, 3), 0.25, 22, ao=0.8)
        # receptáculo
        r = self.esfera * np.array([0.19, 0.17, 0.19], dtype=np.float32)
        c = np.array([0, TALLO_ARRIBA_Y - 0.01, 0], dtype=np.float32)
        nube.agregar(c + r, self.esfera, mezcla(VERDE_OSC, VERDE_CLA, 0.3), 0.2, 16, ao=0.7)

    # ---------------------------------------------------------------- la rosa
    def rosa_malla(self, bloom, escala=1.0, revela=1.0):
        """Superficie espiral única. Devuelve (pos, normales, v, g, mascara)."""
        phi = self.phi
        vv = self.vv
        g = (phi - 0.35) / (self.phi_max - 0.35)                   # 0 centro … 1 afuera
        # los pétalos de afuera se abren primero
        o = suave(0.0, 1.0, (bloom * 1.40 - (1.0 - g) * 0.78) / 0.62)
        r0 = 0.045 + 0.215 * g ** 1.05
        y0 = -0.075 * g
        lp = 0.42 + 0.92 * g ** 0.85
        a0c = 0.05 + 0.40 * g
        afc = -(0.80 + 0.50 * (1 - g))
        a0o = 0.16 + 1.20 * g ** 1.20
        afo = a0o + 0.34 + 0.42 * g
        a0 = a0c + (a0o - a0c) * o
        af = afc + (afo - afc) * o
        k = (af - a0) / lp
        k = np.where(np.abs(k) < 0.03, np.where(k < 0, -0.03, 0.03), k)

        # borde festoneado: lóbulos de cada pétalo
        ph = self.petalos * phi / (2 * np.pi) + self.fase_petalo
        lobulo = np.abs(np.sin(np.pi * ph)) ** 0.55
        festón = 0.58 + 0.42 * lobulo
        s = vv * lp * festón

        ang = a0 + k * s
        rho = r0 + (np.cos(a0) - np.cos(ang)) / k
        yy = y0 + (np.sin(ang) - np.sin(a0)) / k
        # ondulación del borde al abrirse
        ond = np.sin(self.petalos * 2.0 * phi + 4.0 * vv) * 0.060 * vv ** 2 * o
        # cada pétalo es una copa: el centro del lóbulo se abomba y los costados se encorvan
        copa = np.sin(np.pi * (ph % 1.0)) * vv ** 1.2 * lp * (0.20 + 0.10 * o)
        rho = (rho * (1 + ond) + copa * 0.55) * escala * self.tam_flor
        yy = (yy + 0.09 * vv ** 2 * np.sin(self.petalos * phi + 1.7) * o - copa * 0.35) * escala * self.tam_flor

        x = rho * np.cos(phi)
        z = rho * np.sin(phi)
        pos = np.stack([x, yy, z], axis=-1).astype(np.float32)
        nor = normales_malla(pos)
        mascara = (g <= revela) & np.ones_like(vv, dtype=bool)
        return pos, nor, vv + 0 * phi, g + 0 * vv, mascara

    def _rosa_datos(self, bloom, escala, revela):
        pos, nor, v, g, mascara = self.rosa_malla(bloom, escala, revela)
        # color: pétalos de adentro más oscuros; oclusión en la base y en el corazón
        col = mezcla(PETALO_INT, PETALO_EXT, np.clip(g ** 0.8 * 0.8 + v * 0.35, 0, 1))
        ao = (0.46 + 0.54 * suave(0.0, 0.80, v) ** 0.9) * (0.70 + 0.30 * g ** 0.4)
        sel = mascara.reshape(-1)
        rezago = (0.55 * v * (0.4 + 0.6 * g)).reshape(-1)[sel]
        return (pos.reshape(-1, 3)[sel], nor.reshape(-1, 3)[sel], col.reshape(-1, 3)[sel],
                ao.reshape(-1)[sel], rezago)

    def rosa(self, nube, bloom, escala, revela, t):
        if bloom >= 1.0 and escala >= 1.0 and revela >= 1.0:
            # flor abierta: la malla ya no cambia, solo se balancea (ahorra casi todo el cálculo)
            if self._cache_rosa is None:
                self._cache_rosa = self._rosa_datos(1.0, 1.0, 1.0)
            pos, nor, col, ao, rezago = self._cache_rosa
        else:
            pos, nor, col, ao, rezago = self._rosa_datos(bloom, escala, revela)
        nube.agregar(self.balanceo(pos, t, rezago), nor, col, 0.16, 14, ao=ao)

    # ---------------------------------------------------------------- rocío
    def _crear_gotas(self, rng):
        gotas = []
        for i in range(7):
            gotas.append(dict(
                fuente="hoja" if i in (2, 5) else "petalo",
                azim=rng.uniform(0, 2 * np.pi),
                offset=rng.uniform(0, 6.0),
                periodo=rng.uniform(5.2, 7.4),
                radio=rng.uniform(0.040, 0.058),
                vel=rng.uniform(-0.2, 0.2, size=2),
            ))
        return gotas

    def _anclas(self):
        """Puntos del borde de los pétalos externos y de las hojas, con la flor abierta."""
        if self._anclas_flor is not None:
            return self._anclas_flor
        pos, _, _, g, _ = self.rosa_malla(1.0)
        borde = pos[:, -1, :]                    # v = 1: el borde de cada pétalo
        gg = g[:, 0]
        externos = np.flatnonzero(gg > 0.72)
        self._anclas_flor = (borde, externos)
        return self._anclas_flor

    def gotas_rocio(self, nube, t, bloom):
        if bloom < 0.7:
            return
        borde, externos = self._anclas()
        for i, gt in enumerate(self.gotas):
            tau = (t - T_FLORECE_FIN * 0.8 - gt["offset"]) % gt["periodo"]
            crece, cae, salpica = 1.4, 1.25, 0.9
            if gt["fuente"] == "petalo":
                idx = externos[int((gt["azim"] / (2 * np.pi)) * (len(externos) - 1))]
                borde_i = borde[idx].copy()
                a = borde_i
                a = self.balanceo(a[None, :], t - tau)[0]       # se mueve con la planta hasta soltarse
            else:
                a = self._punta_hoja(0 if i == 2 else 1, t - tau)
            if tau < crece:
                r = gt["radio"] * float(suave(0, 1, tau / crece)) ** 0.7
                if gt["fuente"] == "petalo":
                    c = self.balanceo(borde_i[None, :], t)[0] - np.array([0, r * 0.9, 0], dtype=np.float32)
                else:
                    c = self._punta_hoja(0 if i == 2 else 1, t) - np.array([0, r * 0.9, 0], dtype=np.float32)
                self._esfera(nube, c, r, glifo=ord("o"))
            elif tau < crece + cae:
                u = tau - crece
                caida = 4.6 * 0.95
                c = a + np.array([gt["vel"][0] * u, -0.5 * caida * u * u - gt["radio"], gt["vel"][1] * u])
                y_suelo = SUELO_Y + 0.03
                if c[1] > y_suelo:
                    self._esfera(nube, c, gt["radio"] * (1 + 0.15 * u), glifo=ord("O"))
                    # estela que se apaga hacia arriba
                    for j in range(1, 4):
                        cj = c + np.array([0, 0.11 * j, 0])
                        self._esfera(nube, cj, 0.03, brillo=-0.12 * j, glifo=ord("|") if j < 3 else ord(":"))
                else:
                    self._salpicadura(nube, a, gt, 0.0)
            elif tau < crece + cae + salpica:
                self._salpicadura(nube, a, gt, tau - crece - cae)
        # la salpicadura no necesita recalcular el ancla de arriba: usa x,z del punto de caída

    def _punta_hoja(self, k, t):
        """Punta de la hoja k (borde, a ~90 % del largo) con la planta balanceada en el instante t."""
        if self._puntas_hoja is None:
            self._puntas_hoja = []
            for h in self.hojas:
                pos, _ = self._hoja_malla(h, 1.0)
                self._puntas_hoja.append(pos[int(self.nm * 0.90), 0 if h["psi"] > 1 else -1].copy())
        return self.balanceo(self._puntas_hoja[k][None, :], t)[0]

    def _esfera(self, nube, c, r, brillo=0.0, glifo=0):
        nube.agregar(c[None, :] + self.esfera * r, self.esfera, ROCIO, 1.0, 70, ao=1.0, ext=0.35 + brillo,
                     borde=0.0, glifo=glifo)

    def _salpicadura(self, nube, a, gt, u):
        x = a[0] + gt["vel"][0] * (1.05 if u >= 0 else 0)
        z = a[2] + gt["vel"][1] * 1.05
        y = SUELO_Y + 0.03
        # anillo que se expande y se apaga
        fade = max(0.0, 1.0 - u / 0.9)
        n = 54
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
        r = 0.05 + 0.62 * u
        ring = np.stack([x + r * np.cos(ang), np.full(n, y, dtype=np.float32), z + 0.55 * r * np.sin(ang)], axis=1)
        nube.agregar(ring, np.tile([0, 1, 0], (n, 1)), ROCIO, 0.3, 12, ao=0.5, ext=0.40 * fade, borde=0.0,
                     glifo=ord("-") if u < 0.45 else ord("."))
        # gotitas que saltan
        for j in range(9):
            a2 = j * 0.7 + gt["azim"]
            vx, vz = 0.55 * np.cos(a2), 0.45 * np.sin(a2)
            vy = 1.6 + 0.4 * (j % 3)
            p = np.array([x + vx * u, y + vy * u - 0.5 * 5.0 * u * u, z + vz * u], dtype=np.float32)
            if p[1] > y and fade > 0:
                nube.agregar(p[None, :], [[0, 1, 0]], ROCIO, 0.5, 20, ao=1.0, ext=0.25 + 0.35 * fade, borde=0.0,
                             glifo=ord("*") if j % 2 == 0 else ord("'"))

    # ---------------------------------------------------------------- escena completa
    def nube(self, t):
        """Nube de puntos de la planta en el instante t (0 … T_CICLO)."""
        nube = Nube()
        avance = float(np.clip(t / T_CRECE, 0, 1))
        avance = float(suave(0, 1, avance) * 0.65 + avance * 0.35)       # la subida va ligera
        bloom = float(suave(0, 1, (t - T_FLORECE_INI) / (T_FLORECE_FIN - T_FLORECE_INI)))
        yema = float(suave(0, 1, (t - T_YEMA_INI) / (T_YEMA_FIN - T_YEMA_INI)))

        # tallo y hojas: la balanceo se aplica al final a todo salvo la rosa (ya balanceada)
        self._t_actual = t
        tmp = Nube()
        self.tallo(tmp, avance)
        self.sepalos(tmp, float(suave(0, 1, (t - T_CRECE + 0.4) / 1.8)), bloom)
        base = tmp.unir()
        if base is not None:
            p = self.balanceo(base[0], t)
            # normales: se dejan (el doblado es leve)
            for lista, v in zip((nube.p, nube.n, nube.rgb, nube.esp, nube.bri, nube.ao, nube.ext, nube.bor, nube.gli),
                                (p, base[1], base[2], base[3], base[4], base[5], base[6], base[7], base[8])):
                lista.append(v)

        for h in self.hojas:
            self.hoja(nube, h, avance)

        if t > T_YEMA_INI:
            escala = 0.20 + 0.80 * yema ** 0.8
            revela = float(suave(0, 1, yema * 1.15)) * 0.999 + 0.001
            if bloom > 0:
                revela, escala = 1.0, 1.0
            self.rosa(nube, bloom, escala, revela, t)

        self.gotas_rocio(nube, t, bloom)

        # el punto de luz que dibuja el tallo
        if 0.0 < avance < 1.0:
            c = self.balanceo(self.centro_tallo(avance)[None, :], t)[0]
            nube.agregar(c[None, :] + self.esfera * 0.07, self.esfera, LUZ, 0.0, 2, ao=1.0, ext=1.0, borde=0.0)
        return nube.unir(), c if (0.0 < avance < 1.0) else None


# --------------------------------------------------------------------------- fondo de estrellas
class Estrellas:
    def __init__(self, filas, cols, semilla=11):
        rng = np.random.default_rng(semilla)
        n = max(30, int(filas * cols * 0.0125))
        self.y = rng.integers(0, filas, n)
        self.x = rng.integers(0, cols, n)
        self.fase = rng.uniform(0, 6.28, n)
        self.vel = rng.uniform(0.4, 1.6, n)
        self.chars = np.frombuffer(rng.choice(list(b"*+.*+'`"), n).astype(np.uint8).tobytes(), dtype=np.uint8)
        pal = np.array([[235, 235, 245], [235, 235, 245], [255, 130, 170], [130, 175, 255], [255, 220, 150]],
                       dtype=np.float32)
        self.col = pal[rng.integers(0, len(pal), n)]
        self.brillo = rng.uniform(0.35, 1.0, n)
        # destellos grandes y tenues (cruz de líneas)
        m = 6
        self.dy = rng.integers(6, max(7, filas - 6), m)
        self.dx = rng.integers(4, max(5, cols - 4), m)
        self.dfase = rng.uniform(0, 6.28, m)

    def dibujar(self, chars, cols_, t):
        filas, cols = chars.shape
        for i in range(len(self.x)):
            b = (0.5 + 0.5 * math.sin(self.fase[i] + t * self.vel[i])) * self.brillo[i]
            if b < 0.18 or chars[self.y[i], self.x[i]] != 32:
                continue
            chars[self.y[i], self.x[i]] = self.chars[i]
            cols_[self.y[i], self.x[i]] = (self.col[i] * (0.30 + 0.70 * b)).astype(np.uint8)
        for j in range(len(self.dx)):
            b = 0.5 + 0.5 * math.sin(self.dfase[j] + t * 0.45)
            if b < 0.55:
                continue
            y0, x0 = self.dy[j], self.dx[j]
            gris = int(70 * b)
            for dy, dx, ch in ((0, 0, ord("*")), (-1, 0, ord("|")), (1, 0, ord("|")), (0, -2, ord("-")), (0, 2, ord("-")),
                               (0, -3, ord("-")), (0, 3, ord("-")), (-2, 0, ord("|")), (2, 0, ord("|")),
                               (-1, -2, ord("\\")), (1, 2, ord("\\")), (-1, 2, ord("/")), (1, -2, ord("/"))):
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < filas and 0 <= xx < cols and chars[yy, xx] == 32:
                    chars[yy, xx] = ch
                    cols_[yy, xx] = (gris, gris, gris + 8)


# --------------------------------------------------------------------------- renderizador ASCII
class Renderizador:
    def __init__(self, filas, cols, aspecto=1.67, ss=2):
        self.set_tamano(filas, cols, aspecto)
        self.ss = ss
        self.planta = Planta(densidad=self._densidad())
        self.estrellas = Estrellas(filas, cols)

    def _densidad(self):
        return float(np.clip(math.sqrt(self.filas * self.cols / (64 * 120)), 0.45, 1.6))

    def set_tamano(self, filas, cols, aspecto):
        self.filas, self.cols, self.aspecto = filas, cols, aspecto
        self.D = 9.0
        self.K = filas * aspecto * 0.95 * self.D / 3.95

    # ------------------------------------------------------------ cámara
    def _rotacion(self, t):
        yaw = 0.30 * math.sin(self.planta.fase_balanceo(t) * 0.5 + 0.6)      # vaivén suave de la cámara
        pitch = math.radians(29 + 2.5 * math.sin(t * 0.21))
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        return rx @ ry

    # ------------------------------------------------------------ fotograma
    def fotograma(self, t):
        """Devuelve (chars uint8 [filas, cols], colores uint8 [filas, cols, 3], letras)."""
        t = t % (T_CICLO + T_DISUELVE) if t >= 0 else 0.0
        t_sim = min(t, T_CICLO)
        datos, luz = self.planta.nube(t_sim)
        F, C, ss = self.filas, self.cols, self.ss
        chars = np.full((F, C), 32, dtype=np.uint8)
        colores = np.zeros((F, C, 3), dtype=np.uint8)
        letras = 0

        if datos is not None:
            P, N, rgb, esp, shin, ao, ext, bor, gli = datos
            # disolver al final del ciclo
            if t > T_CICLO:
                x = (t - T_CICLO) / T_DISUELVE
                keep = self.planta.umbral[: len(P)] > 1.0 - (1.0 - x) ** 3      # se deshace poco a poco
                P, N, rgb, esp, shin, ao, ext, bor, gli = P[keep], N[keep], rgb[keep], esp[keep], shin[keep], ao[keep], ext[keep], bor[keep], gli[keep]
            R = self._rotacion(t)
            Pc = (P - OBJETIVO) @ R.T
            Nc = N @ R.T
            zc = self.D - Pc[:, 2]
            xs = C * 0.5 + self.K * Pc[:, 0] / zc
            ys = F * 0.50 - self.K * Pc[:, 1] / (zc * self.aspecto)
            ix = np.floor(xs * ss).astype(np.int32)
            iy = np.floor(ys * ss).astype(np.int32)
            ok = (zc > 1.0) & (ix >= 0) & (ix < C * ss) & (iy >= 0) & (iy < F * ss)
            idx = np.flatnonzero(ok)
            if len(idx):
                # luz difusa + especular, a doble cara
                n = Nc[idx]
                n = np.where((n[:, 2] < 0)[:, None], -n, n)
                d = np.clip(n @ LUZ_DIR, 0, 1)
                sp = np.clip(n @ MEDIO, 0, 1) ** shin[idx] * esp[idx]
                lum = (0.28 + 0.72 * d ** 0.9) * ao[idx] + ext[idx]
                lum = np.clip(lum + 0.5 * sp, 0, 1.4)
                lc = np.clip(lum, 0, 1)
                # los brillos se aclaran un poco hacia rosa/verde claro, sin llegar a blanco
                aclara = (np.clip(lc - 0.80, 0, 1) * 1.1)[:, None]
                col = rgb[idx] * (0.30 + 0.70 * lc)[:, None] * (1 + 0.35 * aclara) + 255.0 * (sp * 0.55)[:, None] + 70.0 * aclara
                col = np.clip(col, 0, 255)

                # z-buffer: se asigna de lejos a cerca; gana el último (el más cercano)
                ancho = C * ss
                flat = iy[idx] * ancho + ix[idx]
                orden = np.argsort(-zc[idx], kind="stable")
                buf = np.full(F * ss * ancho, -1, dtype=np.int32)
                buf[flat[orden]] = orden
                sel = buf >= 0
                w = buf[sel]
                sub_lum = np.zeros(F * ss * ancho, dtype=np.float32)
                sub_col = np.zeros((F * ss * ancho, 3), dtype=np.float32)
                sub_z = np.full(F * ss * ancho, 1e9, dtype=np.float32)
                sub_bor = np.zeros(F * ss * ancho, dtype=np.float32)
                sub_gli = np.zeros(F * ss * ancho, dtype=np.uint8)
                sub_lum[sel] = lum[w]
                sub_col[sel] = col[w]
                sub_z[sel] = zc[idx][w]
                sub_bor[sel] = bor[idx][w]
                sub_gli[sel] = gli[idx][w]
                chars, colores, letras = self._celdas(sel, sub_lum, sub_col, sub_z, sub_bor, sub_gli)

        if luz is not None:
            self._halo(chars, colores, luz, R, t)
        self.estrellas.dibujar(chars, colores, t)
        return chars, colores, letras

    # ------------------------------------------------------------ de subceldas a caracteres
    def _celdas(self, sel, sub_lum, sub_col, sub_z, sub_bor, sub_gli):
        F, C, ss = self.filas, self.cols, self.ss
        shape4 = (F, ss, C, ss)
        selr = sel.reshape(shape4).astype(np.float32)
        cuenta = selr.sum((1, 3))
        cov = cuenta / (ss * ss)
        den = np.maximum(cuenta, 1.0)
        lum = (sub_lum.reshape(shape4) * selr).sum((1, 3)) / den
        z4 = sub_z.reshape(shape4).transpose(0, 2, 1, 3).reshape(F, C, ss * ss)
        cerca = z4.argmin(axis=2)
        c4 = sub_col.reshape(F, ss, C, ss, 3).transpose(0, 2, 1, 3, 4).reshape(F, C, ss * ss, 3)
        col = np.take_along_axis(c4, cerca[..., None, None], axis=2)[:, :, 0, :]
        z = z4.min(axis=2)
        con_borde = ((sub_bor.reshape(shape4) * selr).sum((1, 3)) / den) > 0.5
        gl0 = sub_gli.reshape(shape4).max((1, 3))
        lleno = (cov >= 0.20) | ((gl0 > 0) & (cov > 0))

        # contornos: la celda más cercana de un salto de profundidad (o de la silueta)
        lejos = np.where(lleno, z, self.D + 6.0)
        pad = np.pad(lejos, 1, mode="edge")
        izq, der = pad[1:-1, :-2], pad[1:-1, 2:]
        arr, aba = pad[:-2, 1:-1], pad[2:, 1:-1]
        # el umbral de contorno crece con el tamaño de la celda (en terminales pequeñas hay menos celdas)
        salto = 0.085 * (self.D / self.K) / 0.039
        borde = lleno & con_borde & (((izq - z) > salto) | ((der - z) > salto) | ((arr - z) > salto) | ((aba - z) > salto))
        gx = np.clip(der - izq, -4, 4)
        gy = np.clip(aba - arr, -4, 4)

        brillo = np.clip(lum, 0, 1) * (0.55 + 0.45 * np.clip(cov, 0, 1))
        idx = np.clip((brillo * (len(RAMPA) - 1)).astype(np.int32), 1, len(RAMPA) - 1)
        chars = np.where(lleno, RAMPA_ARR[idx], 32).astype(np.uint8)

        # carácter de contorno según la dirección del gradiente de profundidad
        horiz = np.abs(gy) * 1.5 > np.abs(gx)
        vert = np.abs(gx) > np.abs(gy) * 1.5
        diag_barra = (-gy * gx) <= 0
        c_borde = np.where(vert, ord("|"),
                  np.where(horiz, np.where(gy < 0, ord("_"), ord("-")),
                  np.where(diag_barra, ord("/"), ord("\\")))).astype(np.uint8)
        chars = np.where(borde & (brillo > 0.10), c_borde, chars)
        # glifos fijos (gotas, estela, ondas): mandan sobre la rampa
        gl = sub_gli.reshape(shape4).max((1, 3))
        chars = np.where(lleno & (gl > 0), gl, chars).astype(np.uint8)
        colores = np.clip(col * 1.0, 0, 255).astype(np.uint8)
        colores[~lleno] = 0
        letras = int(np.count_nonzero(chars != 32))
        return chars, colores, letras

    # ------------------------------------------------------------ halo del punto de luz
    def _halo(self, chars, colores, c3, R, t):
        Pc = (c3 - OBJETIVO) @ R.T
        zc = self.D - Pc[2]
        cx = int(self.cols * 0.5 + self.K * Pc[0] / zc)
        cy = int(self.filas * 0.5 - self.K * Pc[1] / (zc * self.aspecto))
        for dy in range(-2, 3):
            for dx in range(-4, 5):
                yy, xx = cy + dy, cx + dx
                if not (0 <= yy < self.filas and 0 <= xx < self.cols):
                    continue
                dist = math.hypot(dx / 2.0, dy)
                if dist > 2.3:
                    continue
                f = max(0.0, 1.0 - dist / 2.3)
                ch = ord("@") if dist < 0.6 else ord("*") if dist < 1.3 else ord(".")
                if chars[yy, xx] == 32 or dist < 1.0:
                    chars[yy, xx] = ch
                    colores[yy, xx] = (LUZ * (0.25 + 0.75 * f)).astype(np.uint8)


# --------------------------------------------------------------------------- HUD
def textos_hud(t):
    t = t % (T_CICLO + T_DISUELVE)
    bloom = float(suave(0, 1, (t - T_FLORECE_INI) / (T_FLORECE_FIN - T_FLORECE_INI))) * 100.0
    fase = (t % CICLO_BALANCEO) / CICLO_BALANCEO * 360.0
    return f"t {t:5.2f}s", f"floración {bloom:5.1f}%   balanceo {fase:6.2f}°"


# --------------------------------------------------------------------------- salida: terminal
def a_ansi(chars, colores, color=True):
    filas = []
    for y in range(chars.shape[0]):
        fila = chars[y]
        if not color:
            filas.append(fila.tobytes().decode("ascii"))
            continue
        partes, previo = [], None
        for x in range(chars.shape[1]):
            ch = fila[x]
            if ch == 32:
                partes.append(" ")
                continue
            c = colores[y, x]
            clave = (int(c[0]) >> 3, int(c[1]) >> 3, int(c[2]) >> 3)
            if clave != previo:
                partes.append(f"\x1b[38;2;{clave[0] << 3};{clave[1] << 3};{clave[2] << 3}m")
                previo = clave
            partes.append(chr(ch))
        filas.append("".join(partes) + "\x1b[0m")
    return "\n".join(filas)


def correr_terminal(args):
    try:
        tam = os.get_terminal_size()
        cols, filas = tam.columns, tam.lines - 2
    except OSError:
        cols, filas = 110, 40
    cols, filas = min(cols, 220), max(14, min(filas, 90))
    ren = Renderizador(filas, cols, aspecto=args.aspecto)
    out = sys.stdout
    # con SIGTERM también se restaura la terminal (pantalla alterna y cursor)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")
    t0 = time.perf_counter() - args.inicio
    try:
        while True:
            t = (time.perf_counter() - t0) * args.velocidad
            if not args.bucle and t > T_CICLO:
                t = T_CICLO
            chars, colores, letras = ren.fotograma(t)
            a, b = textos_hud(t)
            out.write("\x1b[H" + a_ansi(chars, colores, not args.sin_color))
            out.write(f"\n\x1b[38;2;255;110;160m● {a}\x1b[0m   \x1b[2m{b}   letras {letras:,}\x1b[0m\x1b[K")
            out.flush()
            time.sleep(max(0.0, 1.0 / args.fps - 0.001))
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[?25h\x1b[?1049l")
        out.flush()


def imprimir_fotograma(args):
    """Un fotograma en texto plano (útil para revisar sin pantalla)."""
    ren = Renderizador(args.filas, args.cols, aspecto=args.aspecto)
    chars, colores, letras = ren.fotograma(args.t)
    print(a_ansi(chars, colores, not args.sin_color))
    a, b = textos_hud(args.t)
    print(f"{a}   {b}   letras {letras:,}")


# --------------------------------------------------------------------------- salida: pygame-ce
class VentanaPygame:
    def __init__(self, ancho, alto, tam_fuente=15, headless=False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        pygame.font.init()
        ruta = (pygame.font.match_font("dejavusansmono,notosansmono,liberationmono,couriernew,menlo,consolas")
                or pygame.font.match_font("monospace"))
        self.fuente = pygame.font.Font(ruta, tam_fuente)
        self.fuente_hud = pygame.font.Font(ruta, max(11, tam_fuente - 2))
        self.cw = self.fuente.size("M")[0]
        self.ch = int(round(self.fuente.get_linesize() * 0.86))
        self.cols, self.filas = max(40, ancho // self.cw), max(20, alto // self.ch)
        self.pantalla = pygame.display.set_mode((self.cols * self.cw, self.filas * self.ch))
        pygame.display.set_caption("Rosa ASCII")
        self.cache = {}

    def glifo(self, ch, color):
        clave = (ch, color[0] >> 3, color[1] >> 3, color[2] >> 3)
        s = self.cache.get(clave)
        if s is None:
            s = self.fuente.render(chr(ch), True, (clave[1] << 3 | 4, clave[2] << 3 | 4, clave[3] << 3 | 4))
            self.cache[clave] = s
        return s

    def dibujar(self, chars, colores, t, letras):
        pg = self.pg
        self.pantalla.fill((2, 2, 4))
        ys, xs = np.nonzero(chars != 32)
        blit = self.pantalla.blit
        for y, x in zip(ys.tolist(), xs.tolist()):
            blit(self.glifo(int(chars[y, x]), colores[y, x].tolist()), (x * self.cw, y * self.ch))
        a, b = textos_hud(t)
        s = self.fuente_hud.render("● " + a, True, (255, 110, 160))
        self.pantalla.blit(s, (self.pantalla.get_width() - s.get_width() - 4 * self.cw, 2 * self.ch))
        gris = (96, 98, 112)
        base = self.pantalla.get_height() - 6 * self.ch
        for i, linea in enumerate(("rosa ASCII · python · numpy · pygame-ce", b, f"letras {letras:,}")):
            self.pantalla.blit(self.fuente_hud.render(linea, True, gris), (4 * self.cw, base + i * int(self.ch * 1.25)))
        pg.display.flip()


def correr_pygame(args):
    try:
        v = VentanaPygame(args.ancho, args.alto)
    except Exception as e:  # sin pantalla (servidor, SSH…)
        sys.exit(f"No se pudo abrir la ventana de pygame ({e}).\n"
                 "Use --modo terminal para verla en la terminal, o --png RUTA para guardar un fotograma.")
    pg = v.pg
    ren = Renderizador(v.filas, v.cols, aspecto=v.ch / v.cw)
    reloj = pg.time.Clock()
    t, vel, pausa = float(args.inicio), args.velocidad, False
    corriendo = True
    while corriendo:
        dt = reloj.tick(args.fps) / 1000.0
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                corriendo = False
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    pausa = not pausa
                elif e.key == pg.K_r:
                    t = 0.0
                elif e.key == pg.K_RIGHT:
                    t += 1.0
                elif e.key == pg.K_LEFT:
                    t = max(0.0, t - 1.0)
                elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                    vel *= 1.25
                elif e.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    vel /= 1.25
        if not pausa:
            t += dt * vel
        if not args.bucle and t > T_CICLO:
            t = T_CICLO
        chars, colores, letras = ren.fotograma(t)
        v.dibujar(chars, colores, t, letras)
    pg.quit()


def guardar_png(args):
    v = VentanaPygame(args.ancho, args.alto, headless=True)
    ren = Renderizador(v.filas, v.cols, aspecto=v.ch / v.cw)
    chars, colores, letras = ren.fotograma(args.t)
    v.dibujar(chars, colores, args.t, letras)
    v.pg.image.save(v.pantalla, args.png)
    print(f"{args.png}  ({v.cols}x{v.filas} celdas, t={args.t:.2f}s, letras {letras:,})")


# --------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser(description="Rosa ASCII animada (Python · NumPy · pygame-ce).")
    ap.add_argument("--modo", choices=["auto", "pygame", "terminal"], default="auto")
    ap.add_argument("--png", metavar="RUTA", help="guarda un fotograma como imagen y termina")
    ap.add_argument("--texto", action="store_true", help="imprime un fotograma en texto y termina")
    ap.add_argument("--t", type=float, default=16.0, help="segundo del fotograma (con --png / --texto)")
    ap.add_argument("--inicio", type=float, default=0.0, help="segundo desde el que arranca la animación")
    ap.add_argument("--velocidad", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--ancho", type=int, default=1080)
    ap.add_argument("--alto", type=int, default=960)
    ap.add_argument("--filas", type=int, default=50, help="solo con --texto")
    ap.add_argument("--cols", type=int, default=100, help="solo con --texto")
    ap.add_argument("--aspecto", type=float, default=2.0, help="alto/ancho de una celda de la terminal")
    ap.add_argument("--sin-color", action="store_true")
    ap.add_argument("--sin-bucle", dest="bucle", action="store_false")
    args = ap.parse_args()

    if args.png:
        guardar_png(args)
    elif args.texto:
        imprimir_fotograma(args)
    else:
        modo = args.modo
        if modo == "auto":
            hay_pantalla = sys.platform in ("win32", "darwin") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
            modo = "pygame" if hay_pantalla else "terminal"
        correr_pygame(args) if modo == "pygame" else correr_terminal(args)


if __name__ == "__main__":
    main()
