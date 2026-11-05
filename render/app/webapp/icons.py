# -*- coding: utf-8 -*-
"""Inline SVG icon set — colored line style (no emojis)."""

ICONS = {
    "brain": ('<path d="M12 5a3 3 0 0 0-3-3 3.5 3.5 0 0 0-3.4 4.2A3.2 3.2 0 0 0 4 12a3.2 3.2 0 0 0 '
              '1.6 5.8A3.5 3.5 0 0 0 9 21a3 3 0 0 0 3-3" stroke="#4da3ff"/>'
              '<path d="M12 5a3 3 0 0 1 3-3 3.5 3.5 0 0 1 3.4 4.2A3.2 3.2 0 0 1 20 12a3.2 3.2 0 0 1 '
              '-1.6 5.8A3.5 3.5 0 0 1 15 21a3 3 0 0 1-3-3" stroke="#38e1c6"/>'),
    "upload": ('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#a78bfa"/>'
               '<polyline points="17 8 12 3 7 8" stroke="#f2c14e"/>'
               '<line x1="12" y1="3" x2="12" y2="15" stroke="#f2c14e"/>'),
    "search": ('<circle cx="11" cy="11" r="8" stroke="#8ec7ff"/>'
               '<line x1="21" y1="21" x2="16.65" y2="16.65" stroke="#38e1c6"/>'),
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" stroke="#f2c14e"/>',
    "chart": ('<path d="M3 3v18h18" stroke="#8fa3bd"/>'
              '<path d="M18 17V9" stroke="#4da3ff"/>'
              '<path d="M13 17V5" stroke="#38e1c6"/>'
              '<path d="M8 17v-3" stroke="#e18fb4"/>'),
    "download": ('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#38e1c6"/>'
                 '<polyline points="7 10 12 15 17 10" stroke="#4da3ff"/>'
                 '<line x1="12" y1="15" x2="12" y2="3" stroke="#4da3ff"/>'),
    "cloud": '<path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" stroke="#8ec7ff"/>',
    "monitor": ('<rect x="2" y="3" width="20" height="14" rx="2" stroke="#a78bfa"/>'
                '<line x1="8" y1="21" x2="16" y2="21" stroke="#8fa3bd"/>'
                '<line x1="12" y1="17" x2="12" y2="21" stroke="#8fa3bd"/>'
                '<path d="M6.5 8h3M11.5 8h2M15 8h2.5" stroke="#38e1c6"/>'),
    "chevron": '<polyline points="9 18 15 12 9 6" stroke="currentColor"/>',
    "package": ('<path d="m7.5 4.27 9 5.15" stroke="#8fa3bd"/>'
                '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 '
                '1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z" stroke="#a78bfa"/>'
                '<path d="m3.3 7 8.7 5 8.7-5" stroke="#4da3ff"/>'
                '<path d="M12 22V12" stroke="#4da3ff"/>'),
    "flask": ('<path d="M10 2v7.5L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9.5V2" stroke="#8ec7ff"/>'
              '<path d="M8.5 2h7" stroke="#8ec7ff"/>'
              '<path d="M7 15h10" stroke="#e18fb4"/>'
              '<path d="M7.5 17.5h9" stroke="#e18fb4"/>'),
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" stroke="#38e1c6"/>',
    "spark": ('<path d="M12 3v3M12 18v3M3 12h3M18 12h3" stroke="#f2c14e"/>'
              '<path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" stroke="#4da3ff"/>'),
    "film": ('<rect x="3" y="5" width="18" height="14" rx="2" stroke="#38e1c6"/>'
             '<path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4" stroke="#4da3ff"/>'),
    "layers": ('<polygon points="12 2 2 7 12 12 22 7 12 2" stroke="#4da3ff"/>'
               '<polyline points="2 17 12 22 22 17" stroke="#38e1c6"/>'
               '<polyline points="2 12 12 17 22 12" stroke="#e18fb4"/>'),
    "file": ('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="#8ec7ff"/>'
             '<polyline points="14 2 14 8 20 8" stroke="#8ec7ff"/>'),
    "link": ('<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" stroke="#4da3ff"/>'
             '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" stroke="#38e1c6"/>'),
}


def render(name, cls=""):
    inner = ICONS.get(name)
    if not inner:
        return ""
    return (f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" '
            f'stroke-linecap="round" stroke-linejoin="round" stroke-width="1.7">'
            f'{inner}</svg>')
