import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class FieldPlotter:
    """
    Clase para crear plots de imágenes, recibe matrices o conjunto de matrices reales o complejas.
    Los plots se generan con matplotlib.pyplot.imshow.
    Es importante que después de usar los métodos de la clase, en una línea se ponga plt.show()
    """
    def __init__(self, dx=1, units = 'pixel', extent = None ):
        """
        dx: mm/pixel (o la unidad que se usa). Valor por defecto 1.
        units: unidad física que poner en los axes.
        extent: (xmin, xmax, ymin, ymax) en unidades físicas; si None, se calcula con dx y shape de la imagen entregada
        """
        self.dx = dx
        self.extent = extent
        self.units = units
    def _compute_extent(self, arr):
        """
        Función para calcular el extent para una imagen (los límites físicos de los ejes de la imagen).
        """
        if self.extent is not None:
            return self.extent
        
        else:
            ny, nx = arr.shape[-2], arr.shape[-1]
            return (-nx*self.dx/2, nx*self.dx/2, -ny*self.dx/2, ny*self.dx/2)

    def plot_intensity(self, field, ax = None, cmap='viridis', norm='linear',
                       vmin=None, vmax=None, show_cb=True, title=None, **imshow_kwargs):
        """Plot |field|^2. Si norm = 'log' se muestra la intensidad en escala logarítmica.
        
        Parameters
        ----------
        field: npdarray NXN
            Matriz con el campo.
        ax: matplotlib plot.
            Si ya el plot está creado, sobreescribe en el plot.
        nomr: string.
            cómo mostrar la escala de los ejes: lineal 'linear', logarítmica 'log'.
        title: string.
            Título de la figura.
        
        Retunrs
        -------
        ax : matplotlib plot.
        """
        
        I = np.abs(field)**2
        if ax is None:
            fig, ax = plt.subplots() # Crea el plot en blanco.
        extent = self._compute_extent(I) # Se calcula el extent.

        if norm == 'log': # Muestra la intensidad en una escala logarítmica.
            im = ax.imshow(I, extent=extent, origin='lower', norm=LogNorm(vmin=max(vmin,1e-12) if vmin else None, vmax=vmax), aspect='equal', cmap=cmap, **imshow_kwargs)
        
        else:
            im = ax.imshow(I, extent=extent, origin='lower', vmin=vmin, vmax=vmax, aspect='equal', cmap=cmap, **imshow_kwargs)
        # Se escriben los labels de los ejes.
        ax.set_xlabel(f"x [{self.units}]")
        ax.set_ylabel(f"y [{self.units}]")
        
        # Se escribe el título.
        if title: ax.set_title(title)
        
        # Muestra la barra de color.
        if show_cb:
            plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
        
        return ax
    def plot_amplitude(self, field, ax = None, cmap='viridis', norm='linear',
                       vmin=None, vmax=None, show_cb=True, title=None, **imshow_kwargs):
        """Plot |field|. Si norm = 'log' se muestra la intensidad en escala logarítmica.
        
        Parameters
        ----------
        field: npdarray NXN
            Matriz con el campo.
        ax: matplotlib plot.
            Si ya el plot está creado, sobreescribe en el plot.
        nomr: string.
            cómo mostrar la escala de los ejes: lineal 'linear', logarítmica 'log'.
        title: string.
            Título de la figura.
        
        Retunrs
        -------
        ax : matplotlib plot.
        """
        
        I = np.abs(field) 
        if ax is None:
            fig, ax = plt.subplots() # Se crea el plot en blanco.
        extent = self._compute_extent(I) # Se calculan el extent.

        if norm == 'log': # Muestra la intensidad en escala logarítmica.
            im = ax.imshow(I, extent=extent, origin='lower', norm=LogNorm(vmin=max(vmin,1e-12) if vmin else None, vmax=vmax), aspect='equal', cmap=cmap, **imshow_kwargs)
        else:
            im = ax.imshow(I, extent=extent, origin='lower', vmin=vmin, vmax=vmax, aspect='equal', cmap=cmap, **imshow_kwargs)
        
        # Se escriben los labels de los ejes.
        ax.set_xlabel(f"x [{self.units}]")
        ax.set_ylabel(f"y [{self.units}]")
        
        # Titula.
        if title: ax.set_title(title)
        
        # Muesra la barra de color.
        if show_cb:
            plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
        
        return ax
    def plot_phase(self, field, ax=None, unwrap=False, cmap='twilight', vmin=-np.pi, vmax=np.pi, show_cb=True, title=None, **imshow_kwargs):
        """Plot arg(Field). unwrap = True sueviza lo entregado opr np.angle.
        
        Parameters
        ----------
        field: npdarray NXN
            Matriz con el campo.
        ax: matplotlib plot.
            Si ya el plot está creado, sobreescribe en el plot.
        title: string.
            Título de la figura.
        
        Retunrs
        -------
        ax : matplotlib plot.
        """
        phi = np.angle(field) # Cálculo de la fase
        
        # unwrap sirve para suavizar los cambios de fase dados por np.angle.
        if unwrap:
            phi = np.unwrap(np.unwrap(phi, axis=1), axis=0)
        if ax is None:
            fig, ax = plt.subplots()
        extent = self._compute_extent(phi) # Se calcula el extent.
        im = ax.imshow(phi, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', **imshow_kwargs)
        
        ax.set_xlabel(f"x [{self.units}]"); ax.set_ylabel(f"y [{self.units}]") # Se escriben los labels de los ejes.

        #Se escribe el título
        if title: ax.set_title(title)
        if show_cb:
            plt.colorbar(im, ax=ax, label='Phase (rad)')
        return ax
    def plot_r_i(self, field, ax=None, cmap='RdBu', titles=('Real','Imag')):
        """Plot de las partes imaginarias y reales de los campos.
        
        Parameters
        ----------
        field: npdarray NXN
            Matriz con el campo.
        ax: matplotlib plot.
            Si ya el plot está creado, sobreescribe en el plot.
        Retunrs
        -------
        ax : matplotlib plot.
        """
        if ax is None:
            fig, axes = plt.subplots(1,2, figsize=(10,4))
        else:
            axes = np.atleast_1d(ax)
            if axes.size < 2:
                raise ValueError("Se necesitan 2 ejes si se pasa 'ax'")
        self.plot_intensity(np.real(field), ax=axes[0], cmap=cmap, show_cb=True, title=titles[0])
        self.plot_intensity(np.imag(field), ax=axes[1], cmap=cmap, show_cb=True, title=titles[1])
        return axes
    
    def plot_all(self, field, figsize=(15, 4), cmap_int='viridis', cmap_phase='twilight',
             norm='linear', vmin_int=None, vmax_int=None, show_cb=True, titles=None):
        """Grafica intensidad, amplitud y fase en una sola figura.
        
        Parameters
        ----------
        field: npdarray NXN
            Matriz con el campo.
        ax: matplotlib plot.
            Si ya el plot está creado, sobreescribe en el plot.
        nomr: string.
            cómo mostrar la escala de los ejes: lineal 'linear', logarítmica 'log'.
        title: string.
            Título de la figura.
        
        Retunrs
        -------
        ax : matplotlib plot.
        """
        
        # Títulos
        if titles is None:
            titles = ['Intensidad', 'Amplitud', 'Fase']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Intensidad
        I = np.abs(field)**2
        extent = self._compute_extent(I) # se calcula el extent
        if norm == 'log':
            im0 = axes[0].imshow(I, extent=extent, origin='lower',
                                norm=LogNorm(vmin=max(vmin_int,1e-12) if vmin_int else None,
                                            vmax=vmax_int),
                                aspect='equal', cmap=cmap_int)
        else:
            im0 = axes[0].imshow(I, extent=extent, origin='lower',
                                vmin=vmin_int, vmax=vmax_int,
                                aspect='equal', cmap=cmap_int)
        axes[0].set_title(titles[0]); axes[0].set_xlabel(f"x [{self.units}]"); axes[0].set_ylabel(f"y [{self.units}]")
        if show_cb: fig.colorbar(im0, ax=axes[0])

        # Amplitud
        A = np.abs(field)
        im1 = axes[1].imshow(A, extent=extent, origin='lower',
                            aspect='equal', cmap=cmap_int)
        axes[1].set_title(titles[1]); axes[1].set_xlabel(f"x [{self.units}]"); axes[1].set_ylabel(f"y [{self.units}]")
        if show_cb: fig.colorbar(im1, ax=axes[1])

        # Fase
        phi = np.angle(field)
        im2 = axes[2].imshow(phi, extent=extent, origin='lower',
                            aspect='equal', cmap=cmap_phase,
                            vmin=-np.pi, vmax=np.pi)
        axes[2].set_title(titles[2]); axes[2].set_xlabel(f"x [{self.units}]"); axes[2].set_ylabel(f"y [{self.units}]")
        if show_cb: fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        return fig, axes
    
    def plot_compare(self, fields, labels, normalize=True, **kwargs):
        """Grafica en la misma figure las amplitudes de los campos entregados. Por defecto normaliza los campos.
        
        Parameters
        ----------
        fields: elemento iterable.
            Elemento iterable que contiene las matrices con los campos.
        normalize: bool.
            Para comparar o no los campos normalizados.
        
        Retunrs
        -------
        ax : matplotlib plot.
        """
        
        fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields),5))

        
        fields = np.array(fields)
        

        if normalize:
            for i in range(len(fields)):
                fields[i] = fields[i] / np.max(np.abs(fields[i]))
        intensities = np.abs(fields)**2

        extent = self._compute_extent(fields[0])

        for i in range(len(fields)):
            im = axes[i].imshow(intensities[i], extent=extent, origin='lower',
                                aspect='equal', cmap='gray', **kwargs)
            axes[i].set_title(labels[i]); axes[i].set_xlabel(f"x [{self.units}]"); axes[0].set_ylabel(f"y [{self.units}]")
            fig.colorbar(im, ax=axes[i])

        plt.tight_layout()
        return fig, axes
    
    def _get_crop(self, field, center=None, size=None, factor=None,
                  center_units='physical', size_units='physical', clip=True):
        """ Hace el recorte de una matriz.

        Parameters
        ----------
        field: numpy array NxN.
            Campo sobre el que se hace el recor.
        center: tuple.
            Posición del centro del recorte, se puede dar en píxeles o unidades
            físicas, especificar en 'center_units'.
        size: tuple o int o float.
            Tamaño del recorte, puede darse en píxeles o unidades físicas.
            Se puede indicar el ancho y largo con una tupla, o un solo valor
            para un recorte cuadrado. Indicar la unidad de la medida en 
            'size_units'.
        factor: float o int.
            Para tomar un recuadro reducido de la imagen, por ejemplo, mitad,
            tercera parte, etc.
        center_units: string
            Unidades de la posición del centro entregado 'physical' para 
            unidad física, cualquier otro valor supone que es entregado en píxeles.
        size_units: string
            Unidades del tamaño del recorte 'physical' para unidad física,
            cualquier otro valor supone que es entregado en píxeles.
        clip: bool
            para proteger el indexado.

        Retunrs
        -------
        crop: recorte del field.
        extent_crop: el extent en unidades físicas para graficación.
        (ix0,ix1,iy0,iy1): posición en pixeles de las dos esquinas del recorte.
        """
        ny, nx = field.shape[-2], field.shape[-1]

        # Determinar centro en píxeles
        # Si no se provee center
        if center is None:
            cx_px = nx // 2
            cy_px = ny // 2
        
        else:
            cx, cy = center
            if center_units == 'physical':
                # extent supuesto: (-nx*dx/2, nx*dx/2, -ny*dx/2, ny*dx/2)
                cx_px = int(round(cx / self.dx + nx / 2))
                cy_px = int(round(cy / self.dx + ny / 2))
            else:  # 'pixels'
                cx_px = int(round(cx))
                cy_px = int(round(cy))

        # Determinar tamaño en píxeles
        if size is not None:
            # Si un solo valor, se asume cuadrado.
            if isinstance(size, (int, float)):
                sx = sy = size
            # Si se da alto y ancho.
            else:
                sx, sy = size
            
            if size_units == 'physical':
                sx_px = int(round(sx / self.dx))
                sy_px = int(round(sy / self.dx))
            else: # 'pixeles'
                sx_px = int(round(sx))
                sy_px = int(round(sy))
        elif factor is not None:
            sx_px = max(1, int(round(nx / float(factor))))
            sy_px = max(1, int(round(ny / float(factor))))
        else:
            # Default: recortar la mitad del tamaño original
            sx_px = nx // 2
            sy_px = ny // 2

        # Asegurar tamaños pares/ímpares razonables
        half_x = sx_px // 2
        half_y = sy_px // 2

        # Indices por los que se va a recortar
        ix0 = cx_px - half_x
        ix1 = cx_px + (sx_px - half_x)
        iy0 = cy_px - half_y
        iy1 = cy_px + (sy_px - half_y)

        if clip: # Por si excede el tamaño de la imagen
            ix0 = max(0, ix0); iy0 = max(0, iy0)
            ix1 = min(nx, ix1); iy1 = min(ny, iy1)

        crop = field[iy0:iy1, ix0:ix1]

        # extent físico del crop: (xmin, xmax, ymin, ymax)
        xmin = (ix0 - nx/2) * self.dx
        xmax = (ix1 - nx/2) * self.dx
        ymin = (iy0 - ny/2) * self.dx
        ymax = (iy1 - ny/2) * self.dx
        extent_crop = (xmin, xmax, ymin, ymax)

        return crop, extent_crop, (ix0,ix1,iy0,iy1)

    def plot_zoom(self, field, center=None, size=None, factor=None,
                  center_units='physical', size_units='physical',
                  resample=False, out_size=None, interp_order=3,
                  ax=None, cmap='viridis', show_cb=True, title=None,
                  vmin=None, vmax=None, **imshow_kwargs):
        """ Hace el recorte de una matriz.

        Parameters
        ----------
        field: numpy array NxN.
            Campo sobre el que se hace el recor.
        center: tuple.
            Posición del centro del recorte, se puede dar en píxeles o unidades
            físicas, especificar en 'center_units'.
        size: tuple o int o float.
            Tamaño del recorte, puede darse en píxeles o unidades físicas.
            Se puede indicar el ancho y largo con una tupla, o un solo valor
            para un recorte cuadrado. Indicar la unidad de la medida en 
            'size_units'.
        factor: float o int.
            Para tomar un recuadro reducido de la imagen, por ejemplo, mitad,
            tercera parte, etc.
        center_units: string
            Unidades de la posición del centro entregado 'physical' para 
            unidad física, cualquier otro valor supone que es entregado en píxeles.
        size_units: string
            Unidades del tamaño del recorte 'physical' para unidad física,
            cualquier otro valor supone que es entregado en píxeles.
        clip: bool
            para proteger el indexado.

        Retunrs
        -------
        crop: recorte del field.
        extent_crop: el extent en unidades físicas para graficación.
        (ix0,ix1,iy0,iy1): posición en pixeles de las dos esquinas del recorte.
        """

        """
        Muestra un zoom (recorte) del campo.
        - center: centro del zoom (x,y) en unidades físicas por defecto. Pon center_units='pixels' para usar índices.
        - size: ancho en unidades (o píxeles si size_units='pixels') o tuple (sx,sy).
        - factor: alternativa a size; el crop será (nx/factor, ny/factor).
        - resample: si True, interpola el crop para devolverlo a tamaño `out_size` (o al tamaño original si out_size None).
                   Usa scipy.ndimage.zoom si está disponible; si no, hace un np.repeat (nearest).
        - out_size: (ny_out, nx_out) en píxeles para la visualización tras re-muestreo. Si None y resample True, usa (ny,nx) original.
        - interp_order: orden de interpolación para scipy.ndimage.zoom (0..5).
        Retorna (ax, crop_array, extent_crop, indices_crop)
        """
        # Halla el tamaño dela imagen
        ny, nx = field.shape[-2], field.shape[-1]

        # Calcula el recorte
        crop, extent_crop, indices = self._get_crop(field, center=center, size=size, factor=factor,
                                                    center_units=center_units, size_units=size_units)
        # decide tamaño de la imagen de salida
        if resample:
            if out_size is None:
                out_ny, out_nx = ny, nx
            else:
                out_ny, out_nx = out_size
            # factor de zoom para llevar crop -> out_size
            zoom_y = out_ny / max(1, crop.shape[0])
            zoom_x = out_nx / max(1, crop.shape[1])
            
            # Si la imagen es más grande, hay que hacer una interpolación en la imagen 
            try:
                from scipy.ndimage import zoom as ndi_zoom
                crop_resampled = ndi_zoom(crop, (zoom_y, zoom_x), order=interp_order)
            except Exception:
                ry = int(round(zoom_y)) if zoom_y >= 1 else 1
                rx = int(round(zoom_x)) if zoom_x >= 1 else 1
                crop_resampled = np.repeat(np.repeat(crop, ry, axis=0), rx, axis=1)
            display_arr = crop_resampled
            # ajustar extent para que muestre las coordenadas físicas correctas del crop original
            extent_display = extent_crop
        else:
            display_arr = crop
            extent_display = extent_crop

        # Graficación

        if ax is None:
            fig, ax = plt.subplots()
        im = ax.imshow(display_arr, origin='lower', extent=extent_display,
                       aspect='equal', cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kwargs)
        ax.set_xlabel(f"x [{self.units}]"); ax.set_ylabel(f"y [{self.units}]")
        if title: ax.set_title(title)
        if show_cb:
            plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
        return ax, crop, extent_crop, indices
