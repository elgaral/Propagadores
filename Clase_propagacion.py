"""
Clase_propagacion.py
Creado el 19 de septiembre de 2022
Autor: elgaral
"""

import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)

class Propa:
    """
    Clase para la propagación de campos ópticos usando aproximaciones paraxiales.
    """
    # Atributos de clase
    version: str = '0.0.2'
    name: str = 'Paraxial-FT'
    units: str = 'Millimeter'

    def __init__(self, L: float = 10.0, N: int = 1024, wl: float = 0.532e-3, z: float = 100.0) -> None:
        """
        Inicializa el objeto de propagación.
        Args:
            L (float): Longitud lateral del área cuadrada (mm).
            N (int): Tamaño lateral de la matriz cuadrada.
            wl (float): Longitud de onda (mm).
            z (float): Distancia de propagación (mm).
        """
        if N <= 0:
            raise ValueError("N debe ser un entero positivo.")
        if L <= 0 or wl <= 0 or z <= 0:
            raise ValueError("L, wl y z deben ser positivos.")
        self.L = float(L)
        self.N = int(N)
        self.wl = float(wl)
        self.z = float(z)
        self.H: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None

    def listar_metodos(self) -> None:
        """
        Lista todos los métodos públicos de la clase y su información básica (primer línea del docstring).
        """
        print('Métodos disponibles en la clase Propa:')
        for attr in dir(self):
            if not attr.startswith('_') and callable(getattr(self, attr)):
                metodo = getattr(self, attr)
                doc = metodo.__doc__
                if doc:
                    resumen = doc.strip().split('\n')[0]
                else:
                    resumen = 'Sin descripción.'
                print(f'- {attr}: {resumen}')

    def parameters(self) -> None:
        """
        Imprime los parámetros actuales del objeto de propagación.
        """
        print('--------------------------------------------------------------')
        print(f'L = {self.L} mm')
        print(f'N = {self.N}')
        print(f'wl = {self.wl} mm')
        print(f'z = {self.z} mm')
        print('--------------------------------------------------------------')

    def resolution(self) -> Tuple[str, float, float]:
        """
        Información sobre la resolución, aproximación sugerida y validez.
        Returns:
            aprox (str): Aproximación sugerida ( Transfer function'TF' o 
             Impulse response 'IR').
            Ds (float): Longitud lateral del área efectiva (mm).
            BW (float): Ancho de banda frecuencia espacial máximo reproducible (1/mm).

        """
        dx = self.L / self.N
        print('--------------------------------------------------------------')
        print('Spatial resolution = L/N \t Regime = wl*z*N/L^2\n')
        print('BW: IR = L/(2*wl*z) \t TF= N/(2*L)')
        print(f'Spatial resolution = {dx} mm')
        regime = self.wl * self.z / (dx * self.L)
        if regime < 1:
            print(f'Regime value = {regime}')
            print('TF approximation suggested')
            aprox = 'TF'
            Ds = self.L / 3 + self.wl * self.z / dx
            BW = 1 / (2 * dx)
        elif regime == 1:
            print(f'Regime value = {regime}')
            print('Ideal case, both approximations work')
            aprox = 'TF'
            Ds = self.L
            BW = 1 / (2 * dx)
        else:
            print(f'Regime value = {regime}')
            print('IR approximation suggested')
            aprox = 'IR'
            Ds = self.L / 3
            BW = self.L / (2 * self.wl * self.z)
        print(f'Side length of effective area = {Ds} mm')
        print(f'Bandwidth = {BW} 1/mm')
        print('--------------------------------------------------------------')
        return aprox, Ds, BW
    
    def TFunc(self, salvar: str = 'N') -> None:
        """
        Crea la función de transferencia para los parámetros del objeto (self.H).
        Args:
            salvar (str): 'Y' para guardar el array en formato npy, 'N' para no guardar.
        """
        dx = self.L / self.N
        k = 2 * np.pi / self.wl
        fx = np.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / self.L)
        FX, FY = np.meshgrid(fx, fx)
        self.H = np.exp(1j * k * self.z) * np.exp(-1j * np.pi * self.wl * self.z * (FX ** 2 + FY ** 2))
        if salvar.upper() == 'Y':
            np.save(f'H_N{self.N}_L{self.L}_wl{self.wl}_z{self.z}', self.H)

    def IRfunc(self, salvar: str = 'N') -> None:
        """
        Crea la función respuesta al impulso para los parámetros del objeto (self.h).
        Args:
            salvar (str): 'Y' para guardar el array en formato npy, 'N' para no guardar.
        """
        dx = self.L / self.N
        k = 2 * np.pi / self.wl
        x = np.arange(-self.L / 2, self.L / 2, dx)
        X, Y = np.meshgrid(x, x)
        self.h = (1 / (1j * self.wl * self.z)) * np.exp(1j * k * self.z) * np.exp(1j * k / (2 * self.z) * (X ** 2 + Y ** 2))
        if salvar.upper() == 'Y':
            np.save(f'h_N{self.N}_L{self.L}_wl{self.wl}_z{self.z}', self.h)

    def propaTF(self, u1: np.ndarray) -> np.ndarray:
        """
        Propaga el objeto usando la aproximación paraxial con el método de función de transferencia.
        Args:
            u1 (np.ndarray): Objeto a propagar (matriz cuadrada).
        Returns:
            np.ndarray: Resultado de la propagación.
        """
        if self.H is None:
            raise ValueError("La función de transferencia H no ha sido calculada. Ejecuta TFunc() primero.")
        if u1.shape[0] != self.N or u1.shape[1] != self.N:
            raise ValueError(f"La matriz de entrada debe ser de tamaño {self.N}x{self.N}.")
        print('--------------------------------------------------------------')
        print(f'propaTF: matríz {self.N} X {self.N}')
        H = np.fft.fftshift(self.H)
        U1 = np.fft.fft2(np.fft.fftshift(u1))
        u2 = np.fft.fftshift(np.fft.ifft2(U1 * H))
        del H, U1
        print('Terminó.')
        print('--------------------------------------------------------------')
        return u2
    
    def propaIR(self, u1: np.ndarray) -> np.ndarray:
        """
        Propaga el objeto usando la aproximación paraxial con el método de respuesta al impulso.
        Args:
            u1 (np.ndarray): Objeto a propagar (matriz cuadrada).
        Returns:
            np.ndarray: Resultado de la propagación.
        """
        if self.h is None:
            raise ValueError("La respuesta al impulso h no ha sido calculada. Ejecuta IRfunc() primero.")
        if u1.shape[0] != self.N or u1.shape[1] != self.N:
            raise ValueError(f"La matriz de entrada debe ser de tamaño {self.N}x{self.N}.")
        print('--------------------------------------------------------------')
        print(f'propaIR: matríz {self.N} X {self.N}')
        dx = self.L / self.N
        H = np.fft.fft2(np.fft.fftshift(self.h)) * dx ** 2
        U1 = np.fft.fft2(np.fft.fftshift(u1))
        u2 = np.fft.fftshift(np.fft.ifft2(U1 * H))
        del H, U1
        print('Terminó.')
        print('--------------------------------------------------------------')
        return u2

            