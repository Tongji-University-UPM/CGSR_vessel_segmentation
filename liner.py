import numpy as np


def _get_hline(st_point, end_point):
    
    m = (end_point[1] - st_point[1])/(end_point[0] - st_point[0])
    b = st_point[1] - st_point[0]*m

    x = np.arange(st_point[0], end_point[0])
    y = np.round(m*x + b).astype(int)

    return x, y

def _get_vline(st_point, end_point):

    st_point, end_point = list(st_point), list(end_point)
    
    end_point[0], end_point[1] = end_point[1], end_point[0]
    st_point[0], st_point[1] = st_point[1], st_point[0]

    m = (end_point[1] - st_point[1])/(end_point[0] - st_point[0])
    b = st_point[1] - st_point[0]*m

    x = np.arange(st_point[0], end_point[0])
    y = np.round(m*x + b).astype(int)

    return y, x

def _filter_rep_points(x_coor, y_coor):

    sep = np.zeros_like(y_coor)

    check_arr = np.char.add(x_coor.astype(str), np.char.add(sep.astype(str), y_coor.astype(str)))

    u, c = np.unique(check_arr, return_counts=True)

    del_ixs = []

    for u_rep in u[c>1]:

        del_ixs.append(np.where(check_arr == u_rep)[0][1:])

    return np.delete(x_coor, np.concatenate(del_ixs)), np.delete(y_coor, np.concatenate(del_ixs))
    

def get_contour(start_points, end_points):

    contour_x, contour_y = [], []

    for xy_s, xy_e in zip(start_points, end_points):

        contour_x.append(xy_s[0])
        contour_y.append(xy_s[1])

        if np.abs(xy_s[0]-xy_e[0])>1 or np.abs(xy_s[1]-xy_e[1])>1:

            if np.abs(xy_s[0]-xy_e[0]) >= np.abs(xy_s[1]-xy_e[1]):

                if xy_s[0]> xy_e[0]:
                    xy_s, xy_e = xy_e, xy_s

                x_int, y_int = _get_hline(xy_s, xy_e)

            else:

                if xy_s[1]> xy_e[1]:
                    xy_s, xy_e = xy_e, xy_s

                x_int, y_int = _get_vline(xy_s, xy_e)

            contour_x.append(x_int)
            contour_y.append(y_int)

    #return _filter_rep_points(np.hstack(contour_x), np.hstack(contour_y))
    return np.hstack(contour_x), np.hstack(contour_y)
