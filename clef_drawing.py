import numpy as np





def _linterp(a, b, n):
    t = np.linspace(0., 1., n)
    return (1.0 - t) * a + t * b


def make_treble(n=100):
    x, y = [], []
    spiral_left_edge = 1.0
    # inner spiral
    theta = np.linspace(np.pi * 1.25, -np.pi, n)
    radius = _linterp(.4, spiral_left_edge, n)
    x.append(np.cos(theta) * radius)
    y.append(np.sin(theta) * radius)
    # sine-up
    slope = np.linspace(1.0, 0, n) ** .4
    t = np.linspace(0.0, np.pi * 1.4, n)
    y.append(np.linspace(0.0, 4.0, n))
    x.append((-spiral_left_edge) * np.cos(t) * (slope))

    # line down
    x.append(np.zeros(n))
    y.append(np.linspace(y[-1][-1], -2.0, n))
    pts = [np.stack((np.concatenate(x),
                     -np.concatenate(y)), axis=1).reshape(-1, 2)]
    return pts, [False]


def make_dot(center, rad, n=10):
    t = np.linspace(0.0, np.pi * 2, n)
    x = center[0] + rad * np.cos(t)
    y = center[1] + rad * np.sin(t)
    return np.dstack((x, y)).reshape(-1, 2)


def make_bass(n=100):
    clef_indent = 0.5
    bass_dot_center_x = 0.5
    bass_dot_radius=.2
    bass_dot_indent_x = 0.55 - bass_dot_radius
    pts = []
    spiral_rad = [0.1, 2.7]
    theta = np.linspace(np.pi , -.75 * np.pi, n)
    scale = np.linspace(0.0, 1.0, n)
    radius = (1.0 - scale) * spiral_rad[0] + scale * spiral_rad[1]
    x = (np.cos(theta) * radius) *.8
    y = (np.sin(theta) * radius)
    rightmost_ind = np.argmax(x)*.8
    topmost_ind = np.argmax(y)
    halfway  =int(y.size/2)
    leftmost_upper_ind = np.argmin(y[:halfway])    
    x[topmost_ind:] -= np.linspace(0.0, 1.4, n-topmost_ind)**3.    
    trim = np.sum( x[halfway:] < x[leftmost_upper_ind])
    x = x[:-(trim-1)]
    y = y[:-(trim-1)]
    x = x - np.max(x) + clef_indent
    pts = [np.stack((x, -y), axis=1).reshape(-1, 2)]
    pts.append(make_dot((bass_dot_indent_x+clef_indent, -0.5), bass_dot_radius))
    pts.append(make_dot((bass_dot_indent_x+clef_indent, 0.5), bass_dot_radius))
    is_closed = (False, True, True)
    return pts, is_closed


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # don't import into app, crashes w/cv2
    plt.subplot(1, 2, 1)
    pts = make_treble()[0]
    for xy in pts:
        plt.plot(xy[:, 0],- xy[:, 1], 'o-')
    plt.axis('equal')
    plt.subplot(1, 2, 2)
    pts = make_bass()[0]
    for xy in pts:
        plt.plot(xy[:, 0],- xy[:, 1], 'o-')
    plt.axis('equal')
    plt.show()
