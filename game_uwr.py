#!/usr/bin/python3.5
"""
 * =====================================================================================================================
 * underwaterrugby 3D simulation application for training purpose
 *   it creates positioning files which can be reloaded for dynamic simulation of UWR games
 *   it creates video for showing movement of players in the swimmingpool field
 *   the matplotlib functionality can be used: 3D rotating, picture saving
 * file name is "game_uwr.py" uploaded in github
 *  make sure the directory /home/family/Bilder exists in your PC or adapt game_uwr.py
      with the new location for storage of video images
 *  put the GUI file into /home/family/glade/game_uwr_180325.glade and/or adapt the name
 *    and/or file location in the script game_uwr.py
 * any date and versions data, see github
 * writing date of this header March30 2018
 * start with "python3 uwr_game.py"
 * Copyright (C) Creative Commons Alike V4.0 https://creativecommons.org/licenses/by-sa/4.0/
 * author pascaldagornet@yahoo.de
 * based on underwaterrugby rules available on vdst.de
 *   and referee training in baden-wuerttemberg by kneer@gmx.net
 * tested/running on Debian9 and the latest available python GTK matplotlib etc. packages
 * GUI designed/created with GLADE
 * No warranty: all sport recommendations/rules of vdst.de remain valid
 * =====================================================================================================================
"""
import math
import time
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk, Gdk as gdk, GLib, GObject as gobject
import os
import subprocess
import glob
import numpy as np
import matplotlib; matplotlib.use('Gtk3Agg')
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import matplotlib.pyplot as plt
import multiprocessing as mp
#
class Annotation3D(Annotation):
    """Annotate the point xyz with text"""
    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)
#
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
#
def draw_basket(ax1, x, y, z, h, color='black'):
    '''add basket to the ax1 figure at the position x y z. basket of height h'''
    # define the basket circle with 16 points
    t = np.linspace(0, np.pi * 2, 16)
    #bottom circle
    ax1.plot(x+0.24*np.cos(t), y+0.24*np.sin(t), z,  linewidth=1, color=color)
    ax1.plot(x+0.16*np.cos(t), y+0.16*np.sin(t), z,  linewidth=1, color=color)
    #top circle
    ax1.plot(x+0.24*np.cos(t), y+0.24*np.sin(t), z+h,  linewidth=1, color=color)
    # 16 side bars
    A=0
    while A < 16:
        xBar = [x+ 0.16 * math.sin(A*22.5*np.pi/180),x+ 0.24 * math.sin(A*22.5*np.pi/180)]
        yBar = [y+ 0.16 * math.cos(A*22.5*np.pi/180),y+ 0.24 * math.cos(A*22.5*np.pi/180)]
        zBar = [0,h]
        ax1.plot(xBar, yBar, zBar, color=color)
        A = A+1

def draw_halfsphere (ax1, x, y, z, sph_radius, color=(0,0,1,1)):
    print("draw halfshere")
    """draw a half of a sphere around the player starting a free
    it show the area where the players of the other team which are inside should not move.
    players of the other team outside that area can attack immediatly the ball holder
    Args:
        ax1 the figure where it will be draw
        x position x of the quarter sphere
        y position y of the quarter sphere
        position z is at the water top =4m in fact
        sph_radius is in fact 2m according the vdst.de rules
    Returns:
        half of sphere as object (can be deleted later)
    """
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi/2:10j]
    xP1 = x + sph_radius * np.cos(u) * np.sin(v)
    yP1 = y + sph_radius * np.sin(u) * np.sin(v)
    zP1 = z - sph_radius * np.cos(v)
    halffreesphere = ax1.plot_wireframe(xP1, yP1, zP1, color=color, alpha=0.3)
    return halffreesphere
#
def draw_quartersphere (ax1, x, y, diam_penalty, side):
    print("draw quarter sphere")
    """draw a quarter of a sphere around the basket representing the penalty
    area the goalkeeper should not leave"
    Args:
        ax1 the figure where it will be draw
        x position x of the quarter sphere
        y position y of the quarter sphere
        position z is at the bottom = 0
        side the penalty is against "b" blue or "w" white
    Returns:
        quarter of sphere as object (can be deleted later)
    """
    if side == "b":
        # sphere blue
        uS, vS = np.mgrid[0:np.pi:20j, 0:np.pi / 2:10j]
    else:
        # sphere white
        uS, vS = np.mgrid[-np.pi:0:20j, 0:np.pi / 2:10j]
    #
    xSphere = x + diam_penalty * np.cos(uS) * np.sin(vS)
    ySphere = y + diam_penalty * np.sin(uS) * np.sin(vS)
    zSphere = diam_penalty * np.cos(vS)
    quartersphere = ax1.plot_wireframe(xSphere, ySphere, zSphere, color='g', alpha=0.3)
    return quartersphere
#
def OnClick(event):
    global selected_coord
    global clicked_coord
    """on click, store the position and calculate the distance to previous clicked position
    if it is the first clicked position, the distance indicated is to 0,0,0
    the output is for now in the window
    Args:
        event
    Output: 
        distance at the screen
    Returns:
        None
    """
#    if function_measurement == "on":
    if foo.button_function_measurement_on.get_active()==True and \
            foo.button_function_measurement_off.get_active()== False:
        clicked_coord [0, 0] = clicked_coord [1, 0]
        clicked_coord [0, 1] = clicked_coord [1, 1]
        clicked_coord [0, 2] = clicked_coord [1, 2]
        clicked_coord [1, 0] = selected_coord[0]
        clicked_coord [1, 1] = selected_coord[1]
        clicked_coord [1, 2] = selected_coord[2]
        print ("selected position X: %5.2f   Y: %5.2f   Z: %5.2f" % (selected_coord[0], selected_coord[1],selected_coord[2]))
        print ("distance to previous selected point:  %5.2f" % np.sqrt ((clicked_coord [0, 0] - clicked_coord [1, 0])**2
                    + (clicked_coord [0, 1]- clicked_coord [1, 1])**2
                    + (clicked_coord [0, 2] - clicked_coord [1, 2])**2))

def distance(point, event):
    """Return distance between mouse position and given data point
    Args:
        point (np.array): np.array of shape (3,), with x,y,z in data coords
        event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
    Returns:
        distance (np.float64): distance (in screen coords) between mouse pos and data point
    """
    # Project 3d data space to 2d data space
    x2, y2, _ = proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
    # Convert 2d data space to 2d screen space
    x3, y3 = ax1.transData.transform((x2, y2))
    return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


def calcClosestDatapoint(X, event):
    """"Calculate which data point is closest to the mouse position.
    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
        event (MouseEvent) - mouse event (containing mouse position)
    returns:
        smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
    """
    distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
    return np.argmin(distances),np.amin(distances)


def annotatePlot(X, index):
    global selected_coord
    global last_mark
    """Create popover label in 3d chart
    Args:
        X (np.array) - array of points, of shape (numPoints, 3)
        index (int) - index (into points array X) of item which should be printed
    Returns:
        None
    """
    # work only if function measurement on in the GUI
    if foo.button_function_measurement_on.get_active():
        # at the on clicking of the function in the GUI, we defined a mark at the ball
        # first it has to be removed
        last_mark.remove()
        x2, y2, _ = proj_transform(X[index, 0], X[index, 1], X[index, 2], ax1.get_proj())
        last_mark = plt.annotate( "Select %d" % (index+1),
            xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        # make coord from label available global for other function like distance measurement  between points
        selected_coord[0]=X[index, 0]
        selected_coord[1]=X[index, 1]
        selected_coord[2]=X[index, 2]
        fig.canvas.draw()

def onMouseMotion(event):
    global pos_pb_now, pos_pw_now
    """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
    if foo.button_function_measurement_on.get_active():
        closestIndexW,LowestDistanceW = calcClosestDatapoint(pos_pw_now, event)
        closestIndexB,LowestDistanceB = calcClosestDatapoint(pos_pb_now, event)
        if LowestDistanceW < LowestDistanceB:
            annotatePlot (pos_pw_now, closestIndexW)
        else:
            annotatePlot (pos_pb_now, closestIndexB)

def OneWindow(s_w_shared,s_d_shared,s_l_shared,el_w_shared,elevation_shared, azimut_shared, pb,
              pw, ball):
    # import the modules because the spawn processing forget the loading at the main process
    import numpy as np
    import matplotlib.pyplot as plt
    #
    ''' Sub-processed Plot viewer of the main windows; copy/paste in one; it helps for PC with 2 monitors
     The main windows remain the control window of the trainer. This window is the view windows of the trained player
     However, the free and penalty sphere are not indicated in this window'''
    #
    def animate_one(i):
        # animation function to just copy the movements of the control window
        p_b_one._offsets3d = pos_pb_now_one[:, 0], pos_pb_now_one[:, 1], pos_pb_now_one[:, 2]
        p_w_one._offsets3d = pos_pw_now_one[:, 0], pos_pw_now_one[:, 1], pos_pw_now_one[:, 2]
        p_ball_one._offsets3d = pos_ball_now_one[:, 0], pos_ball_now_one[:, 1], pos_ball_now_one[:, 2]
        ax1_one.view_init(elev=elevation_shared.value, azim=azimut_shared.value)
    #
    fig_one = plt.figure()
    ax1_one = fig_one.add_subplot(111,projection='3d')
    #
    # access and reshape the shared memory arrays where the positions of the players and ball are
    arrpb = np.frombuffer(pb.get_obj(), dtype='f')
    pos_pb_now_one = np.reshape(arrpb, (6, 3))
    arrpw = np.frombuffer(pw.get_obj(), dtype='f')
    pos_pw_now_one = np.reshape(arrpw, (6, 3))
    arrball = np.frombuffer(ball.get_obj(), dtype='f')
    pos_ball_now_one = np.reshape(arrball, (1, 3))
    #
    # field drawing
    xG = [0,s_w_shared.value,s_w_shared.value,0,0, 0,s_w_shared.value,s_w_shared.value,s_w_shared.value,
          s_w_shared.value,s_w_shared.value, 0, 0,0, 0,s_w_shared.value]
    yG = [0, 0, 0,0,0,s_l_shared.value,s_l_shared.value, 0, 0,s_l_shared.value,s_l_shared.value,s_l_shared.value,
          s_l_shared.value,0,s_l_shared.value,s_l_shared.value]
    zG = [0, 0, s_d_shared.value,s_d_shared.value,0, 0, 0, 0, s_d_shared.value, s_d_shared.value, 0, 0,
          s_d_shared.value,s_d_shared.value, s_d_shared.value, s_d_shared.value]
    ax1_one.plot_wireframe (xG,yG,zG,colors= (0,0,1,1))  # blue line game area
    # exchange area drawing
    xW = [s_w_shared.value,s_w_shared.value+el_w_shared.value,s_w_shared.value+el_w_shared.value,s_w_shared.value,
          s_w_shared.value,s_w_shared.value,s_w_shared.value+el_w_shared.value,s_w_shared.value+el_w_shared.value,
          s_w_shared.value+el_w_shared.value,s_w_shared.value+el_w_shared.value,s_w_shared.value+el_w_shared.value,
          s_w_shared.value,s_w_shared.value,s_w_shared.value,s_w_shared.value,s_w_shared.value+el_w_shared.value]
    yW = [0,  0, 0, 0, 0,s_l_shared.value,s_l_shared.value, 0, 0,s_l_shared.value,s_l_shared.value,s_l_shared.value,
          s_l_shared.value, 0,s_l_shared.value,s_l_shared.value]
    zW = [0,  0, s_d_shared.value, s_d_shared.value, 0, 0, 0, 0, s_d_shared.value, s_d_shared.value, 0, 0,
          s_d_shared.value, s_d_shared.value, s_d_shared.value, s_d_shared.value]
    ax1_one.plot_wireframe (xW,yW,zW,colors= (0,1,1,1))  # light blue line exchange area
    #
    # axis labels
    ax1_one.set_xlabel('Wide')
    ax1_one.set_ylabel('Length')
    ax1_one.set_zlabel('Water')
    #
    # draw the 2 lines at the field bottom which distance to the wall is the depth
    xG1 = [0, s_w_shared.value]
    yG1 = [s_d_shared.value, s_d_shared.value]
    zG1 = [0, 0]
    ax1_one.plot_wireframe(xG1, yG1, zG1, colors=(0, 0, 1, 1),linestyle=':')  # blue line
    xG2 = [0, s_w_shared.value]
    yG2 = [s_l_shared.value-s_d_shared.value, s_l_shared.value-s_d_shared.value]
    zG2 = [0, 0]
    ax1_one.plot_wireframe(xG2, yG2, zG2, colors=(0, 0, 1, 1),linestyle=':')  # blue line
    #
    # put the axis fix
    ax1_one.set_xlim3d(0, s_w_shared.value+el_w_shared.value)
    ax1_one.set_ylim3d(0, s_l_shared.value)
    ax1_one.set_zlim3d(0, s_d_shared.value)
    #
    # use a factor for having y = x in factor
    ax1_one.set_aspect(aspect=0.222)
    #
    # define the basket1
    draw_basket(ax1_one, s_w_shared.value / 2, 0.24, 0., 0.45)
    #
    # define the basket2
    draw_basket(ax1_one, s_w_shared.value / 2, s_l_shared.value - 0.24, 0., 0.45)
    #
    # represent the players with white and blue big balls s=400 can be increased to make them bigger
    p_b_one = ax1_one.scatter(pos_pb_now_one[:, 0], pos_pb_now_one[:, 1], pos_pb_now_one[:, 2],
                          s=400, alpha = 0.5, c=(0, 0, 1, 1))
    p_w_one = ax1_one.scatter(pos_pw_now_one[:, 0], pos_pw_now_one[:, 1],
                      pos_pw_now_one[:, 2], s=400, alpha = 0.5, c="darkgrey")
    p_ball_one = ax1_one.scatter(pos_ball_now_one[:,0], pos_ball_now_one[:,1],
                      pos_ball_now_one[:,2], s=100, alpha = 0.5, c="red")
    #
    # the number of the players are indicated
    for j, xyz_ in enumerate(pos_pb_now_one):
        annotate3D(ax1_one, s=str(j+1), xyz=xyz_, fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right',va='bottom')
    for j, xyz_ in enumerate(pos_pw_now_one):
        annotate3D(ax1_one, s=str(j+1), xyz=xyz_, fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right', va='bottom')
    #
    Frame = 2
    #
    ani1_one = animation.FuncAnimation(fig_one, animate_one, frames=Frame, interval=5, blit=False, repeat=True,
                                       repeat_delay=2)
    #
    plt.pause(0.001)
    plt.show()


def FourWindows(s_w_shared,s_d_shared,s_l_shared,el_w_shared,pb, pw, ball):
    #
    # import the modules because the spawn processing forget the loading at the main process
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as lines
    #
    '''Sub-processed Plot viewer of the main windows; but here in 4 different views = 1x3D + 1xTOP + 1xBack + 1xFront 
    instead of 1x 3D; it helps for PC with 2 monitors
     The main windows remain the control window of the trainer. This window is the view windows for player or 
     referee in a training seminar'''
    #
    def animate_four(i):
        for j in range(6):
            pos_pb_now_2D[j, 0] = pos_pb_now_four[j,1]
            pos_pb_now_2D[j, 1] = s_w_shared.value - pos_pb_now_four[j,0]
            pos_pw_now_2D[j, 0] = pos_pw_now_four[j,1]
            pos_pw_now_2D[j, 1] = s_w_shared.value - pos_pw_now_four[j,0]

        pos_ball_now_2D[0] = pos_ball_now_four[0,1]
        pos_ball_now_2D[1] = s_w_shared.value - pos_ball_now_four[0,0]

        p_b_four1._offsets3d = pos_pb_now_four[:, 0], pos_pb_now_four[:, 1], pos_pb_now_four[:, 2]
        p_b_four2._offsets3d = pos_pb_now_four[:, 0], pos_pb_now_four[:, 1], pos_pb_now_four[:, 2]
        p_b_four4._offsets3d = pos_pb_now_four[:, 0], pos_pb_now_four[:, 1], pos_pb_now_four[:, 2]
        p_b_four3.set_offsets(pos_pb_now_2D)
        p_w_four1._offsets3d = pos_pw_now_four[:, 0], pos_pw_now_four[:, 1],pos_pw_now_four[:, 2]
        p_w_four2._offsets3d = pos_pw_now_four[:, 0], pos_pw_now_four[:, 1],pos_pw_now_four[:, 2]
        p_w_four4._offsets3d = pos_pw_now_four[:, 0], pos_pw_now_four[:, 1],pos_pw_now_four[:, 2]
        p_w_four3.set_offsets(pos_pw_now_2D)
        p_ball_four1._offsets3d = pos_ball_now_four[:, 0], pos_ball_now_four[:, 1], pos_ball_now_four[:, 2]
        p_ball_four2._offsets3d = pos_ball_now_four[:, 0], pos_ball_now_four[:, 1], pos_ball_now_four[:, 2]
        p_ball_four4._offsets3d = pos_ball_now_four[:, 0], pos_ball_now_four[:, 1], pos_ball_now_four[:, 2]
        p_ball_four3.set_offsets(pos_ball_now_2D)
    #
    fig_four = plt.figure()
    #
    # access and reshape the shared memory arrays where the positions of the players and ball are
    arrpb = np.frombuffer(pb.get_obj(), dtype='f')
    pos_pb_now_four = np.reshape(arrpb, (6, 3))
    arrpw = np.frombuffer(pw.get_obj(), dtype='f')
    pos_pw_now_four = np.reshape(arrpw, (6, 3))
    arrball = np.frombuffer(ball.get_obj(), dtype='f')
    pos_ball_now_four = np.reshape(arrball, (1, 3))
    #
    # First subplot SIDE view from exchange side
    ax1_four = fig_four.add_subplot(221, projection='3d')
    ax1_four.view_init(elev=30., azim=10.)
    ax1_four.axis('off')

    # Second subplot Front view from BLUE side
    ax2_four = fig_four.add_subplot(222, projection='3d')
    ax2_four.view_init(elev=30., azim=-90.)
    #
    # Third subplot TOP view of the field; 2D
    ax3_four = fig_four.add_subplot(223)
    ax3_four.grid(True)
    ax3_four.set_aspect("equal")
    ax3_four.axis([0, 18, -1, 10])
    #
    # Fourth subplot Back View from the WHITE side
    ax4_four = fig_four.add_subplot(224, projection='3d')
    ax4_four.view_init(elev=30., azim=90.)
    #
    plt.tight_layout()
    #
    # initiate new array due to the new specific 2D view
    pos_pb_now_2D = []
    pos_pw_now_2D = []
    pos_ball_now_2D = [0.,0.]
    for i in range(6):
        # initiate the array for 2D
        # at the game start 2D view  X2D = Y3D  and  Y2D = s_w minus X3D
        pos_pb_now_2D.append([0., 0.])
        pos_pw_now_2D.append([0., 0.])
    # Define numpy array which is faster to work with
    pos_pb_now_2D = np.array(pos_pb_now_2D, dtype='f')
    pos_pw_now_2D = np.array(pos_pw_now_2D, dtype='f')
    pos_ball_now_2D = np.array(pos_ball_now_2D, dtype='f')
    #
    # field 2D
    xG2D = []
    yG2D = []
    # field 3D
    xG = [0, s_w_shared.value, s_w_shared.value, 0, 0, 0, s_w_shared.value, s_w_shared.value, s_w_shared.value,
          s_w_shared.value, s_w_shared.value, 0, 0, 0, 0, s_w_shared.value]
    yG = [0, 0, 0, 0, 0, s_l_shared.value, s_l_shared.value, 0, 0, s_l_shared.value, s_l_shared.value, s_l_shared.value,
          s_l_shared.value, 0, s_l_shared.value, s_l_shared.value]
    zG = [0, 0, s_d_shared.value, s_d_shared.value, 0, 0, 0, 0, s_d_shared.value, s_d_shared.value, 0, 0,
          s_d_shared.value, s_d_shared.value, s_d_shared.value, s_d_shared.value]
    for i in range(16):
        xG2D.append([yG[i]])
        yG2D.append([s_w_shared.value-xG[i]])
    #
    ax1_four.plot_wireframe(xG, yG, zG, colors=(0, 0, 1, 1))  # blue line game area
    ax2_four.plot_wireframe(xG, yG, zG, colors=(0, 0, 1, 1))  # blue line game area
    line = lines.Line2D(xG2D, yG2D, color=(0, 0, 1, 1))
    ax3_four.add_line(line)
    ax4_four.plot_wireframe(xG, yG, zG, colors=(0, 0, 1, 1))  # blue line game area
    # exchange area 3D
    xW = [s_w_shared.value, s_w_shared.value + el_w_shared.value, s_w_shared.value + el_w_shared.value,
          s_w_shared.value,
          s_w_shared.value, s_w_shared.value, s_w_shared.value + el_w_shared.value,
          s_w_shared.value + el_w_shared.value,
          s_w_shared.value + el_w_shared.value, s_w_shared.value + el_w_shared.value,
          s_w_shared.value + el_w_shared.value,
          s_w_shared.value, s_w_shared.value, s_w_shared.value, s_w_shared.value, s_w_shared.value + el_w_shared.value]
    yW = [0, 0, 0, 0, 0, s_l_shared.value, s_l_shared.value, 0, 0, s_l_shared.value, s_l_shared.value, s_l_shared.value,
          s_l_shared.value, 0, s_l_shared.value, s_l_shared.value]
    zW = [0, 0, s_d_shared.value, s_d_shared.value, 0, 0, 0, 0, s_d_shared.value, s_d_shared.value, 0, 0,
          s_d_shared.value, s_d_shared.value, s_d_shared.value, s_d_shared.value]
    #
    # ExchangeArea 2D for ax3
    xW2D = []
    yW2D = []
    for i in range(16):
        xW2D.append([yW[i]])
        yW2D.append([s_w_shared.value-xW[i]])
    #
    ax1_four.plot_wireframe(xW, yW, zW, colors=(0, 1, 1, 1))  # light blue line exchange area
    ax2_four.plot_wireframe(xW, yW, zW, colors=(0, 1, 1, 1))  # light blue line exchange area
    line = lines.Line2D(xW2D, yW2D, color=(0, 1, 1, 1)) # # light blue line exchange area
    ax3_four.add_line(line)
    ax4_four.plot_wireframe(xW, yW, zW, colors=(0, 1, 1, 1))  # light blue line exchange area
    #
    # draw the 2 lines which show the depth in 3D
    xG1 = [0, s_w_shared.value]
    yG1 = [s_d_shared.value, s_d_shared.value]
    zG1 = [0, 0]
    ax1_four.plot_wireframe(xG1, yG1, zG1, colors=(0, 0, 1, 1), linestyle=':')  # blue line
    # same in 2D view
    xG1_2D = [s_d_shared.value, s_d_shared.value]
    yG1_2D = [s_w_shared.value, 0.]
    line = lines.Line2D(xG1_2D, yG1_2D, color=(0, 0, 1, 1), linestyle=':')
    ax3_four.add_line(line)
    #
    xG2 = [0, s_w_shared.value]
    yG2 = [s_l_shared.value - s_d_shared.value, s_l_shared.value - s_d_shared.value]
    zG2 = [0, 0]
    ax1_four.plot_wireframe(xG2, yG2, zG2, colors=(0, 0, 1, 1), linestyle=':')  # blue line

    xG2_2D = [s_l_shared.value - s_d_shared.value, s_l_shared.value - s_d_shared.value]
    yG2_2D = [s_w_shared.value, 0.]
    line = lines.Line2D(xG2_2D, yG2_2D, color=(0, 0, 1, 1), linestyle=':')
    ax3_four.add_line(line)
    #
    # define the basket1
    draw_basket(ax1_four, s_w_shared.value / 2, 0.24, 0., 0.45)
    draw_basket(ax2_four, s_w_shared.value / 2, 0.24, 0., 0.45)
    t = np.linspace(0, np.pi * 2, 16)
    ax3_four.plot(0.24 + 0.24 * np.sin(t), s_w_shared.value / 2 + 0.24 * np.cos(t), linewidth=1, color='black')
    draw_basket(ax4_four, s_w_shared.value / 2, 0.24, 0., 0.45)
    #
    # define the basket2
    draw_basket(ax1_four, s_w_shared.value / 2, s_l_shared.value - 0.24, 0., 0.45)
    draw_basket(ax2_four, s_w_shared.value / 2, s_l_shared.value - 0.24, 0., 0.45)
    ax3_four.plot(s_l_shared.value - 0.24 + 0.24 * np.sin(t), s_w_shared.value / 2 + 0.24 * np.cos(t)
                  , linewidth=1, color='black')
    draw_basket(ax4_four, s_w_shared.value / 2, s_l_shared.value - 0.24, 0., 0.45)
    #
    # first view
    p_b_four1 = ax1_four.scatter(pos_pb_now_four[:, 0], pos_pb_now_four[:, 1], pos_pb_now_four[:, 2],
                              s=400, alpha=0.5, c=(0, 0, 1, 1))
    p_w_four1 = ax1_four.scatter(pos_pw_now_four[:, 0], pos_pw_now_four[:, 1],
                              pos_pw_now_four[:, 2], s=400, alpha=0.5, c="darkgrey")
    p_ball_four1 = ax1_four.scatter(pos_ball_now_four[:, 0], pos_ball_now_four[:, 1],
                                 pos_ball_now_four[:, 2], s=100, alpha=0.5, c="red")
    #
    for j, xyz_ in enumerate(pos_pb_now_four):
        annotate3D(ax1_four, s=str(j + 1), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    for j, xyz_ in enumerate(pos_pw_now_four):
        annotate3D(ax1_four, s=str(j + 1), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    # second view
    p_b_four2 = ax2_four.scatter(pos_pb_now_four[:, 0], pos_pb_now_four[:, 1], pos_pb_now_four[:, 2],
                              s=400, alpha=0.5, c=(0, 0, 1, 1))
    p_w_four2 = ax2_four.scatter(pos_pw_now_four[:, 0], pos_pw_now_four[:, 1],
                              pos_pw_now_four[:, 2], s=400, alpha=0.5, c="darkgrey")
    p_ball_four2 = ax2_four.scatter(pos_ball_now_four[:, 0], pos_ball_now_four[:, 1],
                                 pos_ball_now_four[:, 2], s=100, alpha=0.5, c="red")
    for j, xyz_ in enumerate(pos_pb_now_four):
        annotate3D(ax2_four, s=str(j + 1), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    for j, xyz_ in enumerate(pos_pw_now_four):
        annotate3D(ax2_four, s=str(j + 1), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    # third view
    p_b_four3 = ax3_four.scatter(pos_pb_now_2D[:, 0], pos_pb_now_2D[:, 1], s=400, alpha=0.5, c=(0, 0, 1, 1))
    p_w_four3 = ax3_four.scatter(pos_pw_now_2D[:, 0], pos_pw_now_2D[:, 1], s=400, alpha=0.5, c="darkgrey")
    p_ball_four3 = ax3_four.scatter(pos_ball_now_2D[0], pos_ball_now_2D[1], s=100, alpha=0.5, c="red")

    for j, xy_ in enumerate(pos_pb_now_2D):
        ax3_four.annotate(s=str(j+1), xy=xy_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    for j, xy_ in enumerate(pos_pw_now_2D):
        ax3_four.annotate(s=str(j + 1), xy=xy_, fontsize=10, xytext=(-3, 3),
                    textcoords='offset points', ha='right', va='bottom')

    # fourth view
    p_b_four4 = ax4_four.scatter(pos_pb_now_four[:, 0], pos_pb_now_four[:, 1], pos_pb_now_four[:, 2],
                              s=400, alpha=0.5, c=(0, 0, 1, 1))
    p_w_four4 = ax4_four.scatter(pos_pw_now_four[:, 0], pos_pw_now_four[:, 1],
                              pos_pw_now_four[:, 2], s=400, alpha=0.5, c="darkgrey")
    p_ball_four4 = ax4_four.scatter(pos_ball_now_four[:, 0], pos_ball_now_four[:, 1],
                                 pos_ball_now_four[:, 2], s=100, alpha=0.5, c="red")
    for j, xyz_ in enumerate(pos_pb_now_four):
        annotate3D(ax4_four, s=str(j + 1), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    for j, xyz_ in enumerate(pos_pw_now_four):
        annotate3D(ax4_four, s=str(j + 1), xyz=xyz_, fontsize=10, xytext=(-3, 3),
                   textcoords='offset points', ha='right', va='bottom')
    #
    Frame = 1
    #
    ani1_one = animation.FuncAnimation(fig_four, animate_four, frames=Frame, interval=1, blit=False, repeat=True,
                                       repeat_delay=1)
    #
    plt.pause(0.001)
    plt.show()

def animate(i):
    global pos_pb_now, pos_pb_now_shared, pos_pb_target, p_b, pos_pb_deltamove
    global pos_pw_now, pos_pw_now_shared, pos_pw_target, p_w, pos_pw_deltamove
    global pos_ball_now, pos_ball_now_shared, pos_ball_target, p_ball, pos_ball_deltamove
    global Frame
    global count_iter
    global video_page_iter
    global azimut_shared
    global elevation_shared
    global player_go_to_clicked
    global move_running
    global lfd_seq
    global dynamic_move_according_file
    global ax1
    global free_sphere
    global frame_divisor
    global anim_video_on
    global animation_slow
    #
    azimut, elevation = ax1.azim, ax1.elev
    azimut_shared.value = azimut
    elevation_shared.value = elevation
    #
    # in case the slow motion is activated, add 100ms at each frame
    if animation_slow:
        plt.pause(0.100)
    #
    if i==0 and player_go_to_clicked:
        player_go_to_clicked = False
        move_running = True
        pos_ball_now[0,:] += (1. / Frame) * pos_ball_deltamove[0,:]
        pos_ball_now_shared[:] = pos_ball_now[0,:]
        pos_pb_now[:,:] += (1. / Frame) * pos_pb_deltamove[:,:]
        pos_pw_now[:,:] += (1. / Frame) * pos_pw_deltamove[:,:]
        pos_pb_now_shared[:] = pos_pb_now.flat[:]
        pos_pw_now_shared[:] = pos_pw_now.flat[:]

        p_b._offsets3d = pos_pb_now[:, 0], pos_pb_now[:, 1], pos_pb_now[:, 2]
        p_w._offsets3d = pos_pw_now[:, 0], pos_pw_now[:, 1], pos_pw_now[:, 2]
        p_ball._offsets3d = pos_ball_now[:, 0], pos_ball_now[:, 1], pos_ball_now[:, 2]

        # store image for future converting to video
        if anim_video_on:
            video_page_iter = video_page_iter + 1
#
########################################################################################################################
########################################################################################################################
# path of the picture directory to be eventually adapted depending of the computer configuration
#
            plt.savefig("/home/family/Bilder" + "/file%03d.png" % video_page_iter)
#
########################################################################################################################
########################################################################################################################
    else:
        if move_running:

            pos_ball_now[0,:] += (1. / Frame) * pos_ball_deltamove[0,:]
            pos_ball_now_shared[:] = pos_ball_now[0,:]
            pos_pb_now += (1. / Frame) * pos_pb_deltamove
            pos_pw_now += (1. / Frame) * pos_pw_deltamove
            pos_pb_now_shared[:] = pos_pb_now.flat[:]
            pos_pw_now_shared[:] = pos_pw_now.flat[:]

            # show only few frames depending of the divisor for the video (no effect on screen)
            if i%frame_divisor == 0:
                p_ball._offsets3d = pos_ball_now[:, 0], pos_ball_now[:, 1], pos_ball_now[:, 2]
                p_b._offsets3d = pos_pb_now[:, 0], pos_pb_now[:, 1], pos_pb_now[:, 2]
                p_w._offsets3d = pos_pw_now[:, 0], pos_pw_now[:, 1], pos_pw_now[:, 2]
                if anim_video_on:
                    video_page_iter = video_page_iter + 1
#
########################################################################################################################
########################################################################################################################
# path of the picture directory to be eventually adapted depending of the computer configuration
#
                    plt.savefig("/home/family/Bilder" + "/file%03d.png" % video_page_iter)
#
########################################################################################################################
########################################################################################################################

            if i == (Frame-1):
                # reset the deltamove to a clean zero for last position in case of rounding elements
                # or set to next step of dynamic move
                pos_ball_deltamove[:,:] = 0.
                pos_ball_now[0,:] = pos_ball_target[0,:]
                pos_ball_now_shared[:] = pos_ball_now[0,:]
                pos_pb_deltamove[:,:] = 0.
                pos_pw_deltamove[:,:] = 0.
                pos_pb_now[:,:] = pos_pb_target[:,:]
                pos_pw_now[:,:] = pos_pw_target[:,:]
                pos_pb_now_shared[:] = pos_pb_now.flat[:]
                pos_pw_now_shared[:] = pos_pw_now.flat[:]

                if dynamic_move_according_file == False:
                    move_running = False    # it indicates the move is at the end. A new move could be after it has been
                                        # activated from the GUI
                else:
                    if lfd_seq < numb_seq and numb_seq!=0 and animation_break == False:
                        lfd_seq += 1

                        pos_pb_target[:,:] = array_coord_sequence[(lfd_seq - 1) * 13:((lfd_seq - 1) * 13)+6,:]
                        pos_pw_target[:,:] = array_coord_sequence[((lfd_seq - 1) * 13)+6:((lfd_seq - 1) * 13)+12,:]
                        pos_pb_deltamove[:,:] = pos_pb_target[:,:] - pos_pb_now[:,:]
                        pos_pw_deltamove[:,:] = pos_pw_target[:,:] - pos_pw_now[:,:]
                        pos_ball_target[0,:] = array_coord_sequence[12 + (lfd_seq - 1) * 13,:]
                        pos_ball_deltamove[0,:] = pos_ball_target[0,:] - pos_ball_now[0,:]
                    else:

                        move_running = False
                        dynamic_move_according_file = False
#
class fooclass:
    """GUI class
    Args:
        none
    Output:
        GUI with several linked functions
    Returns:
        GUI
    """
    #
    def __init__(self):
        #
        # all builder widget and objects
        builder=gtk.Builder()
########################################################################################################################
########################################################################################################################
# path of the GUI file to be eventually adapted depending of the computer configuration
#
        builder.add_from_file("/home/family/glade/game_uwr_180424.glade")
#
########################################################################################################################
########################################################################################################################

        self.window_main = builder.get_object("windows1_uwr_game")
        # coordinates rotation
        self.elevation_scaling = builder.get_object("elevation_scaling")
        self.azimut_scaling = builder.get_object("azimut_scaling")
        #
        # player1 blue
        self.comboboxtext_speed_pb1 = builder.get_object("comboboxtext_speed_pb1")
        self.label_speed_pb1 = builder.get_object("label_speed_pb1")
        self.label_speed_pb1.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_depth_pb1 = builder.get_object("spinbutton_depth_pb1")
        self.label_depth_pb1 = builder.get_object("label_depth_pb1")
        self.label_depth_pb1.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_length_pb1 = builder.get_object("spinbutton_length_pb1")
        self.label_forward_pb1 = builder.get_object("label_forward_pb1")
        self.label_forward_pb1.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_side_pb1 = builder.get_object("spinbutton_side_pb1")
        self.label_side_pb1 = builder.get_object("label_side_pb1")
        self.label_side_pb1.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.eventbox_pb1 = builder.get_object("eventbox_pb1")
        #
        # player2 blue
        self.comboboxtext_speed_pb2 = builder.get_object("comboboxtext_speed_pb2")
        self.label_speed_pb2 = builder.get_object("label_speed_pb2")
        self.label_speed_pb2.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_depth_pb2 = builder.get_object("spinbutton_depth_pb2")
        self.label_depth_pb2 = builder.get_object("label_depth_pb2")
        self.label_depth_pb2.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_length_pb2 = builder.get_object("spinbutton_length_pb2")
        self.label_forward_pb2 = builder.get_object("label_forward_pb2")
        self.label_forward_pb2.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_side_pb2 = builder.get_object("spinbutton_side_pb2")
        self.label_side_pb2 = builder.get_object("label_side_pb2")
        self.label_side_pb2.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.eventbox_pb2 = builder.get_object("eventbox_pb2")
        #
        # player 3 blue
        self.eventbox_pb3 = builder.get_object("eventbox_pb3")
        #
        self.comboboxtext_speed_pb3 = builder.get_object("comboboxtext_speed_pb3")
        self.label_speed_pb3 = builder.get_object("label_speed_pb3")
        self.label_speed_pb3.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_depth_pb3 = builder.get_object("spinbutton_depth_pb3")
        self.label_depth_pb3 = builder.get_object("label_depth_pb3")
        self.label_depth_pb3.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_length_pb3 = builder.get_object("spinbutton_length_pb3")
        self.label_forward_pb3 = builder.get_object("label_forward_pb3")
        self.label_forward_pb3.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_side_pb3 = builder.get_object("spinbutton_side_pb3")
        self.label_side_pb3 = builder.get_object("label_side_pb3")
        self.label_side_pb3.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        #
        # player 4 blue
        self.eventbox_pb4 = builder.get_object("eventbox_pb4")
        self.comboboxtext_speed_pb4 = builder.get_object("comboboxtext_speed_pb4")
        self.label_speed_pb4 = builder.get_object("label_speed_pb4")
        self.label_speed_pb4.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_depth_pb4 = builder.get_object("spinbutton_depth_pb4")
        self.label_depth_pb4 = builder.get_object("label_depth_pb4")
        self.label_depth_pb4.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_length_pb4 = builder.get_object("spinbutton_length_pb4")
        self.label_forward_pb4 = builder.get_object("label_forward_pb4")
        self.label_forward_pb4.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_side_pb4 = builder.get_object("spinbutton_side_pb4")
        self.label_side_pb4 = builder.get_object("label_side_pb4")
        self.label_side_pb4.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        #
        # player 5 blue
        self.eventbox_pb5 = builder.get_object("eventbox_pb5")
        self.comboboxtext_speed_pb5 = builder.get_object("comboboxtext_speed_pb5")
        self.label_speed_pb5 = builder.get_object("label_speed_pb5")
        self.label_speed_pb5.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_depth_pb5 = builder.get_object("spinbutton_depth_pb5")
        self.label_depth_pb5 = builder.get_object("label_depth_pb5")
        self.label_depth_pb5.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_length_pb5 = builder.get_object("spinbutton_length_pb5")
        self.label_forward_pb5 = builder.get_object("label_forward_pb5")
        self.label_forward_pb5.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_side_pb5 = builder.get_object("spinbutton_side_pb5")
        self.label_side_pb5 = builder.get_object("label_side_pb5")
        self.label_side_pb5.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        self.spinbutton_side_pb5 = builder.get_object("spinbutton_side_pb5")
        self.label_depth_pb5 = builder.get_object("label_depth_pb5")
        self.label_depth_pb5.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))
        #
        # player 6 blue
        self.eventbox_pb6 = builder.get_object("eventbox_pb6")
        self.comboboxtext_speed_pb6 = builder.get_object("comboboxtext_speed_pb6")
        self.label_speed_pb6 = builder.get_object("label_speed_pb6")
        self.label_speed_pb6.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_depth_pb6 = builder.get_object("spinbutton_depth_pb6")
        self.label_depth_pb6 = builder.get_object("label_depth_pb5")
        self.label_depth_pb6.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_length_pb6 = builder.get_object("spinbutton_length_pb6")
        self.label_forward_pb6 = builder.get_object("label_forward_pb6")
        self.label_forward_pb6.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_side_pb6 = builder.get_object("spinbutton_side_pb6")
        self.label_side_pb6 = builder.get_object("label_side_pb6")
        self.label_side_pb6.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        self.spinbutton_side_pb6 = builder.get_object("spinbutton_side_pb6")
        self.label_depth_pb6 = builder.get_object("label_depth_pb6")
        self.label_depth_pb6.override_color(gtk.StateFlags.NORMAL, gdk.RGBA(1, 1, 1, 1))

        # player1 white
        self.comboboxtext_speed_pw1 = builder.get_object("comboboxtext_speed_pw1")
        self.spinbutton_depth_pw1 = builder.get_object("spinbutton_depth_pw1")
        self.spinbutton_length_pw1 = builder.get_object("spinbutton_length_pw1")
        self.spinbutton_side_pw1 = builder.get_object("spinbutton_side_pw1")
        #
        # player 2 white
        self.comboboxtext_speed_pw2 = builder.get_object("comboboxtext_speed_pw2")
        self.spinbutton_depth_pw2 = builder.get_object("spinbutton_depth_pw2")
        self.spinbutton_length_pw2 = builder.get_object("spinbutton_length_pw2")
        self.spinbutton_side_pw2 = builder.get_object("spinbutton_side_pw2")
        #
        # player 3 white
        self.comboboxtext_speed_pw3 = builder.get_object("comboboxtext_speed_pw3")
        self.spinbutton_depth_pw3 = builder.get_object("spinbutton_depth_pw3")
        self.spinbutton_length_pw3 = builder.get_object("spinbutton_length_pw3")
        self.spinbutton_side_pw3 = builder.get_object("spinbutton_side_pw3")
        #
        # player 4 white
        self.comboboxtext_speed_pw4 = builder.get_object("comboboxtext_speed_pw4")
        self.spinbutton_depth_pw4 = builder.get_object("spinbutton_depth_pw4")
        self.spinbutton_length_pw4 = builder.get_object("spinbutton_length_pw4")
        self.spinbutton_side_pw4 = builder.get_object("spinbutton_side_pw4")
        #
        # player 5 white
        self.comboboxtext_speed_pw5 = builder.get_object("comboboxtext_speed_pw5")
        self.spinbutton_depth_pw5 = builder.get_object("spinbutton_depth_pw5")
        self.spinbutton_length_pw5 = builder.get_object("spinbutton_length_pw5")
        self.spinbutton_side_pw5 = builder.get_object("spinbutton_side_pw5")
        #
        # player 6 white
        self.comboboxtext_speed_pw6 = builder.get_object("comboboxtext_speed_pw6")
        self.spinbutton_depth_pw6 = builder.get_object("spinbutton_depth_pw6")
        self.spinbutton_length_pw6 = builder.get_object("spinbutton_length_pw6")
        self.spinbutton_side_pw6 = builder.get_object("spinbutton_side_pw6")
        #
        # ball
        self.spinbutton_depth_ball = builder.get_object("spinbutton_depth_ball")
        self.spinbutton_length_ball = builder.get_object("spinbutton_length_ball")
        self.spinbutton_side_ball = builder.get_object("spinbutton_side_ball")
        #
        # diverse button
        self.button_go_to = builder.get_object("button_go_to")
        #
        self.button_anim_video_on = builder.get_object("button_anim_video_on")
        self.button_anim_video_pause = builder.get_object("button_anim_video_pause")
        self.button_anim_video_off = builder.get_object("button_anim_video_off")
        #
        self.button_animation_on = builder.get_object("button_animation_on")
        self.button_animation_off = builder.get_object("button_animation_off")
        self.button_animation_break = builder.get_object("button_animation_break")
        #
        # generate move combobox
        self.move_combobox = builder.get_object("move_combobox")
        #
        # generate ball pos combobox
        self.ball_combobox_pos = builder.get_object("ball_combobox_pos")
        # generate ball player color combobox
        self.ball_combobox_playercol = builder.get_object("ball_combobox_playercol")
        # generate ball player number combobox
        self.ball_combobox_playernb = builder.get_object("ball_combobox_playernb")
        #
        # generate 1/2 sphere free
        self.activate_free_sphere_on = builder.get_object("button_free_sphere_on")
        self.activate_free_sphere_off = builder.get_object("button_free_sphere_off")
        self.free_sphere_pos_middle = builder.get_object("free_sphere_pos_middle")
        self.free_sphere_pos_frontblue = builder.get_object("free_sphere_pos_frontblue")
        self.free_sphere_pos_frontwhite = builder.get_object("free_sphere_pos_frontwhite")
        self.free_sphere_pos_player = builder.get_object("free_sphere_pos_player")
        self.free_combobox_playercol = builder.get_object("free_combobox_playercol")
        self.free_combobox_playernb = builder.get_object("free_combobox_playernb")
        #
        # generate penalty 1/4 sphere
        self.activate_penalty_sphere_on = builder.get_object("button_penalty_sphere_on")
        self.activate_penalty_sphere_off = builder.get_object("button_penalty_sphere_off")
        self.penalty_sphere_side_blue = builder.get_object("penalty_sphere_side_blue")
        self.penalty_sphere_side_white = builder.get_object("penalty_sphere_side_white")
        #
        # function measurement buttons
        self.button_function_measurement_on = builder.get_object("button_function_measurement_on")
        self.button_function_measurement_off = builder.get_object("button_function_measurement_off")
        #
        # function add viewer windows
        self.add_separate_one_window_on = builder.get_object("button_add_separate_one_window_on")
        self.add_separate_one_window_off = builder.get_object("button_add_separate_one_window_off")
        self.add_separate_four_window_on = builder.get_object("button_add_separate_four_window_on")
        self.add_separate_four_window_off = builder.get_object("button_add_separate_four_window_off")
        #
        self.animation_globalspeedstandard = builder.get_object("button_animation_globalspeedstandard")
        self.animation_globalspeedslow = builder.get_object("button_animation_globalspeedslow")
        #
        # file treatment entries
        self.default_filename_coord_store = builder.get_object("entry_filename_coord_store")
        self.write_file_coord = builder.get_object("ChooserButton_write_file_coord")
        self.read_file_coord = builder.get_object("ChooserButton_read_file_coord")
        self.frame_scaling = builder.get_object("scale_frame")
        self.default_filename_video_store = builder.get_object("entry_filename_video_store")
        #
        self.suptitle_text = builder.get_object("entry_suptitle_text")
        self.label_active_pos = builder.get_object("label_active_pos")
        #
        builder.connect_signals(self)
        #
        # update_eventbox of blue players to blue
        # pb1 to pb6
        self.eventbox_pb1.override_background_color(gtk.StateFlags.NORMAL, gdk.RGBA(0, 0, 1, 1))
        self.eventbox_pb2.override_background_color(gtk.StateFlags.NORMAL, gdk.RGBA(0, 0, 1, 1))
        self.eventbox_pb3.override_background_color(gtk.StateFlags.NORMAL, gdk.RGBA(0, 0, 1, 1))
        self.eventbox_pb4.override_background_color(gtk.StateFlags.NORMAL, gdk.RGBA(0, 0, 1, 1))
        self.eventbox_pb5.override_background_color(gtk.StateFlags.NORMAL, gdk.RGBA(0, 0, 1, 1))
        self.eventbox_pb6.override_background_color(gtk.StateFlags.NORMAL, gdk.RGBA(0, 0, 1, 1))

    #
    def player_go_to(self, widget, data=None):
        global s_w
        global s_d
        global s_l
        global pos_pb_now
        global pos_pw_now
        global pos_ball_now
        global pos_pb_deltamove
        global pos_pw_deltamove
        global pos_ball_deltamove
        global pos_pb_target
        global pos_pw_target
        global pos_ball_target
        global player_go_to_clicked
        global move_running
        global dynamic_move_according_file
        global filename_coord_store
        global filename_coord_retrieve
        global lfd_seq
        global animation_slow
        global plot_suptitle
        global plot_suptitle_string
        #
        '''by clicking the GO TO button in the GUI, all target move will be readen'''
        #
        # identify what type of move is set in the menue
        identified_move = str(self.move_combobox.get_active_text())
        identified_move_ball = str(self.ball_combobox_pos.get_active_text())
        identified_move_ball_playercol = str(self.ball_combobox_playercol.get_active_text())
        identified_move_ball_playernb = str(self.ball_combobox_playernb.get_active_text())
        identified_move_ball_playernb = int(identified_move_ball_playernb)
        #
        # default move of the ball as delta to any player is zero
        dx_pos_ball = 0.
        dy_pos_ball = 0.
        dz_pos_ball = 0.
        #
        plot_suptitle.remove()
        plot_suptitle = ax1.text2D(0., 1., plot_suptitle_string, fontweight='bold', fontsize=15,
                                   transform=ax1.transAxes,
                                   bbox={'facecolor': 'lightgreen', 'alpha': 0.5, 'pad': 10})
        #
        # get at what speed the animation will have to be done
        if self.animation_globalspeedslow.get_active():
            animation_slow = True
        if self.animation_globalspeedstandard.get_active():
            animation_slow = False
        #
        if identified_move == "to menu coord":

            print("players to go to coordinates")
            dynamic_move_according_file = False

            # get all data of the GUI
            # dont allow a new start of move when one is running
            if move_running == False:

                pos_pb_target[0, 2] = s_d - self.spinbutton_depth_pb1.get_value()
                pos_pb_target[0, 1] = self.spinbutton_length_pb1.get_value()
                pos_pb_target[0, 0] = self.spinbutton_side_pb1.get_value()

                pos_pb_target[1, 2] = s_d - self.spinbutton_depth_pb2.get_value()
                pos_pb_target[1, 1] = self.spinbutton_length_pb2.get_value()
                pos_pb_target[1, 0] = self.spinbutton_side_pb2.get_value()

                pos_pb_target[2, 2] = s_d - self.spinbutton_depth_pb3.get_value()
                pos_pb_target[2, 1] = self.spinbutton_length_pb3.get_value()
                pos_pb_target[2, 0] = self.spinbutton_side_pb3.get_value()

                pos_pb_target[3, 2] = s_d - self.spinbutton_depth_pb4.get_value()
                pos_pb_target[3, 1] = self.spinbutton_length_pb4.get_value()
                pos_pb_target[3, 0] = self.spinbutton_side_pb4.get_value()

                pos_pb_target[4, 2] = s_d - self.spinbutton_depth_pb5.get_value()
                pos_pb_target[4, 1] = self.spinbutton_length_pb5.get_value()
                pos_pb_target[4, 0] = self.spinbutton_side_pb5.get_value()

                pos_pb_target[5, 2] = s_d - self.spinbutton_depth_pb6.get_value()
                pos_pb_target[5, 1] = self.spinbutton_length_pb6.get_value()
                pos_pb_target[5, 0] = self.spinbutton_side_pb6.get_value()
                #
                pos_pw_target[0, 2] = s_d - self.spinbutton_depth_pw1.get_value()
                pos_pw_target[0, 1] = s_l - self.spinbutton_length_pw1.get_value()
                pos_pw_target[0, 0] = s_w - self.spinbutton_side_pw1.get_value()

                pos_pw_target[1, 2] = s_d - self.spinbutton_depth_pw2.get_value()
                pos_pw_target[1, 1] = s_l - self.spinbutton_length_pw2.get_value()
                pos_pw_target[1, 0] = s_w - self.spinbutton_side_pw2.get_value()

                pos_pw_target[2, 2] = s_d - self.spinbutton_depth_pw3.get_value()
                pos_pw_target[2, 1] = s_l - self.spinbutton_length_pw3.get_value()
                pos_pw_target[2, 0] = s_w - self.spinbutton_side_pw3.get_value()

                pos_pw_target[3, 2] = s_d - self.spinbutton_depth_pw4.get_value()
                pos_pw_target[3, 1] = s_l - self.spinbutton_length_pw4.get_value()
                pos_pw_target[3, 0] = s_w - self.spinbutton_side_pw4.get_value()

                pos_pw_target[4, 2] = s_d - self.spinbutton_depth_pw5.get_value()
                pos_pw_target[4, 1] = s_l - self.spinbutton_length_pw5.get_value()
                pos_pw_target[4, 0] = s_w - self.spinbutton_side_pw5.get_value()

                pos_pw_target[5, 2] = s_d - self.spinbutton_depth_pw6.get_value()
                pos_pw_target[5, 1] = s_l - self.spinbutton_length_pw6.get_value()
                pos_pw_target[5, 0] = s_w - self.spinbutton_side_pw6.get_value()
                #
                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

                player_go_to_clicked = True # only after the coordinates were set

        elif identified_move == "all back to wall":

            print("all players to go to start position at the wall")

            dynamic_move_according_file = False

            # the players move back to the start position. The spinbutton are not updated

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = 1.0   # blue 1m distance to wall
                pos_pb_target[i, 2] = s_d
                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = s_l - 1.0
                pos_pw_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "blue back to wall":

            print("players blue to go to start position at the wall; white stays")

            dynamic_move_according_file = False

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = 0.5   # blue 1m distance to wall
                pos_pb_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "blue to middle top":

            print("players blue to go to middle top; white stays")

            dynamic_move_according_file = False

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = s_l/2.0
                pos_pb_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "blue to mid&bottom":

            print("players blue to go to middle bottom; white stays")
            dynamic_move_according_file = False

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = s_l / 2.0
                pos_pb_target[i, 2] = 0.5

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "blue forward top":

            print("players blue to go to forward top; white stays")
            dynamic_move_according_file = False

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = s_l - 0.5
                pos_pb_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "blue front free top":

            print("players blue to go to forward top at free distance of 3m from the wall; white stays")
            dynamic_move_according_file = False

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = s_l - 3.0
                pos_pb_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "blue forward bottom":

            print("players blue to go to forward bottom; white stays")
            dynamic_move_according_file = False

            for i in range(6):
                # distribute the players at the side with the same distance
                # at game start
                pos_pb_target[i, 0] = ((s_w / 6) / 2) + i * (s_w / 6)
                pos_pb_target[i, 1] = s_l - 0.5
                pos_pb_target[i, 2] = 0.5

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "white back to wall":

            print("players white to go to start position at the wall; blue stays")

            dynamic_move_according_file = False

            for i in range(6):

                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = s_l - 0.5
                pos_pw_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):

                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "white to middle top":

            print("players white to go to middle top; blue stays")
            dynamic_move_according_file = False

            for i in range(6):

                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = s_l / 2.0
                pos_pw_target[i, 2] = s_d

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):

                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "white to mid&bottom":

            print("players white to go to middle bottom; blue stays")
            dynamic_move_according_file = False

            for i in range(6):

                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = s_l / 2.0
                pos_pw_target[i, 2] = 0.5

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):

                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "white forward top":

            print("players white to go to forward top; blue stays")

            dynamic_move_according_file = False

            for i in range(6):

                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = 0.5
                pos_pw_target[i, 2] = s_d

            if not move_running:    # delta move calculation only when a move is not activ

                for j in range(6):

                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "white forward bottom":

            print("players white to go to forward bottom; blue stays")
            dynamic_move_according_file = False

            for i in range(6):

                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = 0.5
                pos_pw_target[i, 2] = 0.5

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):

                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "white front free top":

            print("players white to go to forward at free distance 3m from wall; blue stays")

            dynamic_move_according_file = False

            for i in range(6):

                pos_pw_target[i, 0] = s_w - ((s_w / 6) / 2) - i * (s_w / 6)
                pos_pw_target[i, 1] = 3.0
                pos_pw_target[i, 2] = s_d

            if not move_running:    # delta move calculation only when a move is not activ

                for j in range(6):

                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set


        elif  identified_move == "penalty against white":

            dynamic_move_according_file = False

            # the players move back to the penalty start position. The ball stay where it was
            # player1 blue in the middle
            pos_pb_target[0, 0] = 5.0
            pos_pb_target[0, 1] = 9.0
            pos_pb_target[0, 2] = 4.0
            # player1 white over the basket
            pos_pw_target[0, 0] = 5.0
            pos_pw_target[0, 1] = 17.2
            pos_pw_target[0, 2] = 4.0

            for i in range(1,6):
                # distribute the players at the exchange line
                # at game start
                pos_pw_target[i, 0] = 10.5
                pos_pw_target[i, 1] = 17.5 - (i-1)*0.8
                pos_pw_target[i, 2] = 4.
                pos_pb_target[i, 0] = 10.5
                pos_pb_target[i, 1] = 13.5 - (i-1)*0.8
                pos_pb_target[i, 2] = 4

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "penalty against blue":
            dynamic_move_according_file = False

            # the players move back to the penalty start position. The ball stay where it was
            # player1 white in the middle
            pos_pw_target[0, 0] = 5.0
            pos_pw_target[0, 1] = 9.0
            pos_pw_target[0, 2] = 4.0
            # player1 blue over the basket
            pos_pb_target[0, 0] = 5.0
            pos_pb_target[0, 1] = 0.8
            pos_pb_target[0, 2] = 4.0

            for i in range(1,6):
                # distribute the players at the exchange line
                # at game start
                pos_pw_target[i, 0] = 10.5
                pos_pw_target[i, 1] = 4.5 + (i-1)*0.8
                pos_pw_target[i, 2] = 4.
                pos_pb_target[i, 0] = 10.5
                pos_pb_target[i, 1] = 0.5 + (i-1)*0.8
                pos_pb_target[i, 2] = 4

            if move_running == False:    # delta move calculation only when a move is not activ

                for j in range(6):
                    pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                    pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                    pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                    pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                    pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                    pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

            player_go_to_clicked = True # only after the coordinates were set


        if move_running == False:
            if identified_move_ball == "player mid":

                dx_pos_ball = 0.
                dy_pos_ball = 0.
                dz_pos_ball = 0.

            elif identified_move_ball == "player front":

                dx_pos_ball = 0.
                dy_pos_ball = 0.6
                dz_pos_ball = 0.

            elif identified_move_ball == "player back":

                dx_pos_ball = 0.
                dy_pos_ball = -0.6
                dz_pos_ball = 0.

            elif identified_move_ball == "player left":

                dx_pos_ball = -0.4
                dy_pos_ball = 0.
                dz_pos_ball = 0.

            elif identified_move_ball == "player right":

                dx_pos_ball = 0.4
                dy_pos_ball = 0.
                dz_pos_ball = 0.

            elif identified_move_ball == "player top":

                dx_pos_ball = 0.
                dy_pos_ball = 0.
                dz_pos_ball = 0.3

            elif identified_move_ball == "player down":

                dx_pos_ball = 0.
                dy_pos_ball = 0.
                dz_pos_ball = -0.3


            if identified_move_ball_playercol == "white":

                # inverse X and Y in case of white player instead of blue
                dx_pos_ball = 0. - dx_pos_ball
                dy_pos_ball = 0. - dy_pos_ball
                dz_pos_ball = 0.

                pos_ball_target[0, 0] = pos_pw_target[identified_move_ball_playernb - 1, 0] + dx_pos_ball
                pos_ball_target[0, 1] = pos_pw_target[identified_move_ball_playernb - 1, 1] + dy_pos_ball
                pos_ball_target[0, 2] = pos_pw_target[identified_move_ball_playernb - 1, 2] + dz_pos_ball

            else:

                pos_ball_target[0, 0] = pos_pb_target[identified_move_ball_playernb - 1, 0] + dx_pos_ball
                pos_ball_target[0, 1] = pos_pb_target[identified_move_ball_playernb - 1, 1] + dy_pos_ball
                pos_ball_target[0, 2] = pos_pb_target[identified_move_ball_playernb - 1, 2] + dz_pos_ball


            if identified_move_ball == "1. middle bottom":

                pos_ball_target[0, 0] = 5.
                pos_ball_target[0, 1] = 9.
                pos_ball_target[0, 2] = 0.2

            elif identified_move_ball == "middle top":

                pos_ball_target[0, 0] = 5.
                pos_ball_target[0, 1] = 9.
                pos_ball_target[0, 2] = 4.

            elif identified_move_ball == "free front blue":

                pos_ball_target[0, 0] = 5.
                pos_ball_target[0, 1] = 15.
                pos_ball_target[0, 2] = 4.

            elif identified_move_ball == "free front white":

                pos_ball_target[0, 0] = 5.
                pos_ball_target[0, 1] = 3.
                pos_ball_target[0, 2] = 4.

            elif identified_move_ball == "basket blue":

                pos_ball_target[0, 0] = 5.
                pos_ball_target[0, 1] = 0.24
                pos_ball_target[0, 2] = 0.2

            elif identified_move_ball == "basket white":

                pos_ball_target[0, 0] = 5.
                pos_ball_target[0, 1] = s_l_shared.value - 0.24
                pos_ball_target[0, 2] = 0.2

            elif identified_move_ball == "coordinate":

                pos_ball_target[0, 0] = self.spinbutton_side_ball.get_value()
                pos_ball_target[0, 1] = self.spinbutton_length_ball.get_value()
                pos_ball_target[0, 2] = s_d - self.spinbutton_depth_ball.get_value()


            pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
            pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
            pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]

        if  identified_move == "acc all seq from file":

            print("players go to positions according file sequence; number of identified sequences: ", numb_seq)
            dynamic_move_according_file = True

            if numb_seq > 0:   # non empty file was opened then initialize the first position

                lfd_seq = 1
                for i in range(6):
                    # use first the top of the array from file
                    pos_pb_target[i, 0] = array_coord_sequence [i, 0]
                    pos_pb_target[i, 1] = array_coord_sequence [i, 1]
                    pos_pb_target[i, 2] = array_coord_sequence [i, 2]
                    pos_pw_target[i, 0] = array_coord_sequence [i+6, 0]
                    pos_pw_target[i, 1] = array_coord_sequence [i+6, 1]
                    pos_pw_target[i, 2] = array_coord_sequence [i+6, 2]
                pos_ball_target[0, 0] = array_coord_sequence [12, 0]
                pos_ball_target[0, 1] = array_coord_sequence [12, 1]
                pos_ball_target[0, 2] = array_coord_sequence [12, 2]

                if move_running == False:    # delta move calculation only when a move is not activ

                    for j in range(6):
                        pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                        pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                        pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                        pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                        pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                        pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

                    pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
                    pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
                    pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]
                    #
            player_go_to_clicked = True  # only after the coordinates were set

        elif identified_move == "till end of file":

            dynamic_move_according_file = True
            #
            if lfd_seq < numb_seq:  # move only if it was not already at the end of the file
                lfd_seq = lfd_seq + 1
                print("players go till end of file sequence; ", lfd_seq)

                buffer_label_active_pos = "%03d" % lfd_seq
                self.label_active_pos.set_text(buffer_label_active_pos)

                for i in range(6):
                    # use the top of the array from file
                    pos_pb_target[i, 0] = array_coord_sequence[i + (lfd_seq - 1) * 13, 0]
                    pos_pb_target[i, 1] = array_coord_sequence[i + (lfd_seq - 1) * 13, 1]
                    pos_pb_target[i, 2] = array_coord_sequence[i + (lfd_seq - 1) * 13, 2]
                    pos_pw_target[i, 0] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 0]
                    pos_pw_target[i, 1] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 1]
                    pos_pw_target[i, 2] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 2]
                pos_ball_target[0, 0] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 0]
                pos_ball_target[0, 1] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 1]
                pos_ball_target[0, 2] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 2]

                if move_running == False:  # delta move calculation only when a move is not activ.. before pb_target calc?

                    for j in range(6):
                        pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                        pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                        pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                        pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                        pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                        pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

                    pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
                    pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
                    pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]

                player_go_to_clicked = True  # only after the coordinates were set

        elif  identified_move == "to first file pos":

            print("players go to first position according file sequence")
            dynamic_move_according_file = False
            #
            if numb_seq > 0:   # non empty file was opened
                lfd_seq = 1

                buffer_label_active_pos = "%03d" % lfd_seq
                self.label_active_pos.set_text(buffer_label_active_pos)

                for i in range(6):
                    # initialize the top of the array from file
                    pos_pb_target[i, 0] = array_coord_sequence [i, 0]
                    pos_pb_target[i, 1] = array_coord_sequence [i, 1]
                    pos_pb_target[i, 2] = array_coord_sequence [i, 2]
                    pos_pw_target[i, 0] = array_coord_sequence [i+6, 0]
                    pos_pw_target[i, 1] = array_coord_sequence [i+6, 1]
                    pos_pw_target[i, 2] = array_coord_sequence [i+6, 2]
                pos_ball_target[0, 0] = array_coord_sequence [12, 0]
                pos_ball_target[0, 1] = array_coord_sequence [12, 1]
                pos_ball_target[0, 2] = array_coord_sequence [12, 2]

                if move_running == False:    # delta move calculation only when a move is not activ

                    for j in range(6):
                        pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                        pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                        pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                        pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                        pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                        pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]
                    #
                    pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
                    pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
                    pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]

                player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "to last file pos":

            print("players go to last position according file sequence; ", numb_seq)
            dynamic_move_according_file = False
            #
            if numb_seq > 0:   # non empty file was opened
                lfd_seq = numb_seq
            #
                buffer_label_active_pos = "%03d" % lfd_seq
                self.label_active_pos.set_text(buffer_label_active_pos)

                for i in range(6):
                # use the top of the array from file
                    pos_pb_target[i, 0] = array_coord_sequence [i+(lfd_seq - 1) * 13, 0]
                    pos_pb_target[i, 1] = array_coord_sequence [i+(lfd_seq - 1) * 13, 1]
                    pos_pb_target[i, 2] = array_coord_sequence [i+(lfd_seq - 1) * 13, 2]
                    pos_pw_target[i, 0] = array_coord_sequence [i+(lfd_seq - 1) * 13+6, 0]
                    pos_pw_target[i, 1] = array_coord_sequence [i+(lfd_seq - 1) * 13+6, 1]
                    pos_pw_target[i, 2] = array_coord_sequence [i+(lfd_seq - 1) * 13+6, 2]
                pos_ball_target[0, 0] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 0]
                pos_ball_target[0, 1] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 1]
                pos_ball_target[0, 2] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 2]

                if move_running == False:    # delta move calculation only when a move is not activ.. before pb_target calc?

                    for j in range(6):
                        pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                        pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                        pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                        pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                        pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                        pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]
                    pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
                    pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
                    pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]

                player_go_to_clicked = True # only after the coordinates were set

        elif  identified_move == "to next file pos":

            dynamic_move_according_file = False
            #
            if lfd_seq < numb_seq:    # move only if it was not already at the end of the file
                lfd_seq = lfd_seq + 1

                buffer_label_active_pos = "%03d" % lfd_seq
                self.label_active_pos.set_text(buffer_label_active_pos)

                print("players go to next position according file sequence; ", lfd_seq)
                for i in range(6):
                    # use the top of the array from file
                    pos_pb_target[i, 0] = array_coord_sequence[i + (lfd_seq - 1) * 13, 0]
                    pos_pb_target[i, 1] = array_coord_sequence[i + (lfd_seq - 1) * 13, 1]
                    pos_pb_target[i, 2] = array_coord_sequence[i + (lfd_seq - 1) * 13, 2]
                    pos_pw_target[i, 0] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 0]
                    pos_pw_target[i, 1] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 1]
                    pos_pw_target[i, 2] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 2]
                pos_ball_target[0, 0] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 0]
                pos_ball_target[0, 1] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 1]
                pos_ball_target[0, 2] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 2]

                if move_running == False:  # delta move calculation only when a move is not activ.. before pb_target calc?

                    for j in range(6):
                        pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                        pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                        pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                        pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                        pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                        pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

                    pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
                    pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
                    pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]

                player_go_to_clicked = True  # only after the coordinates were set

        elif  identified_move == "to previous file pos":

            dynamic_move_according_file = False

            if lfd_seq > 1:  # move only if it was not already at the beginning of the file

                lfd_seq = lfd_seq - 1

                buffer_label_active_pos = "%03d" % lfd_seq
                self.label_active_pos.set_text(buffer_label_active_pos)

                print("players go to previous position according file sequence; ", lfd_seq)

                for i in range(6):
                    # use the top of the array from file
                    pos_pb_target[i, 0] = array_coord_sequence[i + (lfd_seq - 1) * 13, 0]
                    pos_pb_target[i, 1] = array_coord_sequence[i + (lfd_seq - 1) * 13, 1]
                    pos_pb_target[i, 2] = array_coord_sequence[i + (lfd_seq - 1) * 13, 2]
                    pos_pw_target[i, 0] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 0]
                    pos_pw_target[i, 1] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 1]
                    pos_pw_target[i, 2] = array_coord_sequence[i + (lfd_seq - 1) * 13 + 6, 2]
                pos_ball_target[0, 0] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 0]
                pos_ball_target[0, 1] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 1]
                pos_ball_target[0, 2] = array_coord_sequence[12 + (lfd_seq - 1) * 13, 2]

                if move_running == False:  # delta move calculation only when a move is not activ.. before pb_target calc?

                    for j in range(6):
                        pos_pb_deltamove[j, 0] = pos_pb_target[j, 0] - pos_pb_now[j, 0]
                        pos_pb_deltamove[j, 1] = pos_pb_target[j, 1] - pos_pb_now[j, 1]
                        pos_pb_deltamove[j, 2] = pos_pb_target[j, 2] - pos_pb_now[j, 2]
                        pos_pw_deltamove[j, 0] = pos_pw_target[j, 0] - pos_pw_now[j, 0]
                        pos_pw_deltamove[j, 1] = pos_pw_target[j, 1] - pos_pw_now[j, 1]
                        pos_pw_deltamove[j, 2] = pos_pw_target[j, 2] - pos_pw_now[j, 2]

                    pos_ball_deltamove[0, 0] = pos_ball_target[0, 0] - pos_ball_now[0, 0]
                    pos_ball_deltamove[0, 1] = pos_ball_target[0, 1] - pos_ball_now[0, 1]
                    pos_ball_deltamove[0, 2] = pos_ball_target[0, 2] - pos_ball_now[0, 2]

                player_go_to_clicked = True  # only after the coordinates were set


        elif  identified_move == "tbd..":
            player_go_to_clicked = False

    #
    #
    def set_azimut_elevation(self, widget, data=None):
        """set the azimut and elevation to GUI values
        Args:
            elev and azim
        Output:
            updated ax1
        Returns:
            None
        """
        azimut = self.azimut_scaling.get_value()
        elevation = self.elevation_scaling.get_value()
        ax1.view_init(elev=elevation, azim=azimut)
        print("azimut elevation new ", azimut, elevation)
    #
    #
#    def change_settings(self, widget, data=None):
#        global filename_coord_store
#        global animation_slow
#        print("change settings")
        # file name
#        filename_coord_store = self.default_filename_coord_store.get_text()
#        if filename_coord_store == "":
#            filename_coord_store = 'pos_uwr_player.csv'
#            self.default_filename_coord_store.set_text('pos_uwr_player.csv')
#        else:
#            filename_coord_store = filename_coord_store + '.csv'
#
    def animation_standard_on (self, widget, data=None):
        global animation_slow
        if self.animation_globalspeedstandard.get_active():
            print("animation change to standard")
            animation_slow = False
    #
    def animation_slow_on (self, widget, data=None):
        global animation_slow
        if self.animation_globalspeedslow.get_active():
            print("animation change to slow")
            animation_slow = True
    #
    def on_game_uwr_exit(self,widget,data=None):
        print("exit")
        plt.close('all')
        gtk.main_quit()

    def set_coord_menue (self, widget, data=None):
        """reset all coord data in the spinbutton of the GUI according the current position"""
        print("set menue coord data to current player and ball pos in game view")
        if move_running == False:
            #
            self.spinbutton_depth_pb1.set_value(s_d-pos_pb_now[0, 2])
            self.spinbutton_length_pb1.set_value(pos_pb_now[0, 1])
            self.spinbutton_side_pb1.set_value(pos_pb_now[0, 0])
            #
            self.spinbutton_depth_pb2.set_value(s_d-pos_pb_now[1, 2])
            self.spinbutton_length_pb2.set_value(pos_pb_now[1, 1])
            self.spinbutton_side_pb2.set_value(pos_pb_now[1, 0])
            #
            self.spinbutton_depth_pb3.set_value(s_d-pos_pb_now[2, 2])
            self.spinbutton_length_pb3.set_value(pos_pb_now[2, 1])
            self.spinbutton_side_pb3.set_value(pos_pb_now[2, 0])
            #
            self.spinbutton_depth_pb4.set_value(s_d-pos_pb_now[3, 2])
            self.spinbutton_length_pb4.set_value(pos_pb_now[3, 1])
            self.spinbutton_side_pb4.set_value(pos_pb_now[3, 0])
            #
            self.spinbutton_depth_pb5.set_value(s_d-pos_pb_now[4, 2])
            self.spinbutton_length_pb5.set_value(pos_pb_now[4, 1])
            self.spinbutton_side_pb5.set_value(pos_pb_now[4, 0])
            #
            self.spinbutton_depth_pb6.set_value(s_d-pos_pb_now[5, 2])
            self.spinbutton_length_pb6.set_value(pos_pb_now[5, 1])
            self.spinbutton_side_pb6.set_value(pos_pb_now[5, 0])
            #
            self.spinbutton_depth_pw1.set_value(s_d-pos_pw_now[0, 2])
            self.spinbutton_length_pw1.set_value(s_l-pos_pw_now[0, 1])
            self.spinbutton_side_pw1.set_value(s_w-pos_pw_now[0, 0])
            #
            self.spinbutton_depth_pw2.set_value(s_d-pos_pw_now[1, 2])
            self.spinbutton_length_pw2.set_value(s_l-pos_pw_now[1, 1])
            self.spinbutton_side_pw2.set_value(s_w-pos_pw_now[1, 0])
            #
            self.spinbutton_depth_pw3.set_value(s_d-pos_pw_now[2, 2])
            self.spinbutton_length_pw3.set_value(s_l-pos_pw_now[2, 1])
            self.spinbutton_side_pw3.set_value(s_w-pos_pw_now[2, 0])
            #
            self.spinbutton_depth_pw4.set_value(s_d-pos_pw_now[3, 2])
            self.spinbutton_length_pw4.set_value(s_l-pos_pw_now[3, 1])
            self.spinbutton_side_pw4.set_value(s_w-pos_pw_now[3, 0])
            #
            self.spinbutton_depth_pw5.set_value(s_d-pos_pw_now[4, 2])
            self.spinbutton_length_pw5.set_value(s_l-pos_pw_now[4, 1])
            self.spinbutton_side_pw5.set_value(s_w-pos_pw_now[4, 0])
            #
            self.spinbutton_depth_pw6.set_value(s_d-pos_pw_now[5, 2])
            self.spinbutton_length_pw6.set_value(s_l-pos_pw_now[5, 1])
            self.spinbutton_side_pw6.set_value(s_w-pos_pw_now[5, 0])
            #
            self.spinbutton_depth_ball.set_value(s_d - pos_ball_now[0, 2])
            self.spinbutton_length_ball.set_value(pos_ball_now[0, 1])
            self.spinbutton_side_ball.set_value(pos_ball_now[0, 0])
#
    def store_coord (self, widget, data=None):
        global pos_pb_now
        global pos_pw_now
        global pos_ball_now
        global move_running
        global filename_coord_store
        global f_store_handler
        print("store coord into file")
        # if move not running (= player at the target position)
        # if file not open, open it then append the pos blue and white to it
        if move_running == False:
            #
            if self.write_file_coord.get_filename() == None:

                if self.default_filename_coord_store.get_text() == "":
                    filename_coord_store = 'pos_uwr_player.csv'
                else:
                    filename_coord_store = filename_coord_store + '.csv'
                print("store coord into file NEW", filename_coord_store)
                f_store_handler = open(filename_coord_store, 'wb')

            else:
                filename_coord_store = self.write_file_coord.get_filename()
                print("store coord into existing file ", filename_coord_store)
                f_store_handler = open(filename_coord_store, 'ab')
            #
            np.savetxt(f_store_handler, pos_pb_now, fmt='%5.2f', delimiter=' , ')
            np.savetxt(f_store_handler, pos_pw_now, fmt='%5.2f', delimiter=' , ')
            np.savetxt(f_store_handler, pos_ball_now, fmt='%5.2f', delimiter=' , ')
            f_store_handler.close()  # close after at each opening else lost of data by exiting
            #

    def update_pos_seq (self, widget, data=None):
        global pos_pb_now
        global pos_pw_now
        global pos_ball_now
        global move_running
        global filename_coord_store
        global f_store_handler
        print("update current uploaded file seq according data in the screen")
        # if move not running (= player at the target position)
        # if file not open, open it then append the pos blue and white to it
        if move_running == False:
            #
            if numb_seq !=0:

                for k in range(6):
                    array_coord_sequence[k + (lfd_seq - 1) * 13, 0] = pos_pb_now[k, 0]
                    array_coord_sequence[k + (lfd_seq - 1) * 13, 1] = pos_pb_now[k, 1]
                    array_coord_sequence[k + (lfd_seq - 1) * 13, 2] = pos_pb_now[k, 2]
                    array_coord_sequence[k + (lfd_seq - 1) * 13 + 6, 0] = pos_pw_now[k, 0]
                    array_coord_sequence[k + (lfd_seq - 1) * 13 + 6, 1] = pos_pw_now[k, 1]
                    array_coord_sequence[k + (lfd_seq - 1) * 13 + 6, 2] = pos_pw_now[k, 2]

                array_coord_sequence[12 + (lfd_seq - 1) * 13, 0] = pos_ball_now[0, 0]
                array_coord_sequence[12 + (lfd_seq - 1) * 13, 1] = pos_ball_now[0, 1]
                array_coord_sequence[12 + (lfd_seq - 1) * 13, 2] = pos_ball_now[0, 2]

            #
    def store_file_coord (self, widget, data=None):
        global move_running
        global filename_coord_store
        print("store the sequences again into the file")
        # if move not running (= player at the target position)
        # if file not open, open it then append the pos blue and white to it
        if move_running == False:
            #
            filename_coord_store = self.default_filename_coord_store.get_text()
            #
            if filename_coord_store == "":
                filename_coord_store = 'pos_uwr_player.csv'
            else:
                if filename_coord_store.find('.csv') == -1:  # -1 will be returned when a is not in b
                    filename_coord_store = filename_coord_store + '.csv'
            print("store coord into file new/overwriting", filename_coord_store)
            #
            f_store_handler_local = open(filename_coord_store, 'wb')
            #
            np.savetxt(f_store_handler_local, array_coord_sequence, fmt='%5.2f', delimiter=' , ')
            f_store_handler_local.close()  # close after at each opening else lost of data by exiting

            #
    #
    def write_file_activated (self, widget, data=None):
        global filename_coord_store
        filename_coord_store = self.write_file_coord.get_filename()
        print("function write_file_activated")
        print("choosen file to store coord", filename_coord_store)
    #
    def retrieve_file_coord  (self, widget, data=None):
        global filename_coord_retrieve
        global array_coord_sequence
        global numb_seq
        filename_coord_retrieve = self.read_file_coord.get_filename()
#        print("choosen file to retrieve coord: ", filename_coord_retrieve)
        array_coord_sequence = np.loadtxt(filename_coord_retrieve, delimiter=',', skiprows=0, dtype='f')
#        print("array uploaded: ", array_coord_sequence)
        length_array, wide_array = array_coord_sequence.shape
#        print("array length: ", length_array)
#        print("array wide: ", wide_array)
        numb_seq, whatever = divmod(length_array, 13)
        print("number identified sequences: ", numb_seq)
    #
    def insert_pos_before (self, widget, data=None):
        global array_coord_sequence
        global numb_seq
        global lfd_seq

        if numb_seq !=0:
            # add 13 lines distributed in 1 sequence
            print("insert new sequence before (only in memory)  ", lfd_seq)
            length_array, wide_array = array_coord_sequence.shape
            print("array before insertion length x wide : ", length_array, wide_array)
            b = np.copy(array_coord_sequence)
            array_coord_sequence = array_coord_sequence.copy()
            array_coord_sequence.resize(((numb_seq + 1)*13, 3))
            array_coord_sequence[lfd_seq*13:,:]= b [(lfd_seq-1)*13:,:]
            numb_seq += 1
            length_array, wide_array = array_coord_sequence.shape
            print("array after insertion length x wide : ", length_array, wide_array)

    #
    def delete_this_pos (self, widget, data=None):
        global array_coord_sequence
        global numb_seq
        global lfd_seq
        if numb_seq > 0:
            print("delete this file sequence (only in memory)  ", lfd_seq)
            length_array, wide_array = array_coord_sequence.shape
            print("array before deletion length x wide : ", length_array, wide_array)

#            b = np.delete(array_coord_sequence,[range((lfd_seq-1)*13,lfd_seq*13)],0)
            b = np.copy(array_coord_sequence)
            array_coord_sequence = array_coord_sequence.copy()
            array_coord_sequence.resize(((numb_seq - 1)*13, 3))
            array_coord_sequence[(lfd_seq-1)*13:,:]= b [lfd_seq*13:,:]

            numb_seq -=1
            length_array, wide_array = array_coord_sequence.shape
            print("array after deletion length x wide : ", length_array, wide_array)
    #
    def free_sphere_on (self, widget, data=None):
        global free_sphere
        global pos_pb_now, pos_pw_now
        """
        function free_sphere generated from GUI
        Args:
        - from GUI
        - global pos_pb_now pos_pw_now numpy array of the player positions
        Out:
        - global free_sphere object simulating the free area around the player
        Return:
        - none
        """
        # generate free
        if (self.activate_free_sphere_on.get_active()):
            print("free sphere activation")
            if (self.free_sphere_pos_middle.get_active()):
                free_sphere = draw_halfsphere(ax1, 5., 9., 4., 2.)
            elif (self.free_sphere_pos_frontblue.get_active()):
                free_sphere = draw_halfsphere(ax1, 5., 15., 4., 2.)
            elif (self.free_sphere_pos_frontwhite.get_active()):
                free_sphere = draw_halfsphere(ax1, 5., 3., 4., 2.)
            elif(self.free_sphere_pos_player.get_active()):
                index_free_player = int(self.free_combobox_playernb.get_active_text())
                if (self.free_combobox_playercol.get_active_text()=="blue"):
                    free_sphere = draw_halfsphere(ax1, pos_pb_now[index_free_player-1,0],
                                                  pos_pb_now[index_free_player-1,1], 4., 2.)
                else:
                    free_sphere = draw_halfsphere(ax1, pos_pw_now[index_free_player-1,0],
                                                  pos_pw_now[index_free_player-1,1], 4., 2.)

    #
    def free_sphere_off (self, widget, data=None):
        global free_sphere
        """
        function free_sphere deactivated from GUI
        Args:
        - from GUI
        - global free_sphere which will be removed
        Out:
        - none
        Return:
        - none
        """
        if (self.activate_free_sphere_off.get_active()):
            print("free sphere deletion")
            free_sphere.remove()
            fig.canvas.draw()
    #
    def penalty_sphere_on (self, widget, data=None):
        global penalty_sphere
        """
        function penalty_sphere generated from GUI
        Args:
        - from GUI
        Out:
        - global penalty_sphere object simulating the penalty area around the goal keeper
        Return:
        - none
        """
        if (self.activate_penalty_sphere_on.get_active()):
            print("penalty sphere activation")
            if (self.penalty_sphere_side_blue.get_active()):
                penalty_sphere = draw_quartersphere(ax1, 5., 0., 2.5, "b")
            elif (self.penalty_sphere_side_white.get_active()):
                penalty_sphere = draw_quartersphere(ax1, 5., 18., 2.5, "w")
    #
    def penalty_sphere_off (self, widget, data=None):
        global penalty_sphere
        """
        function penalty_sphere deactivated from GUI
        Args:
        - from GUI
        - global penalty_sphere which will be removed
        Out:
        - none
        Return:
        - none
        """
        if (self.activate_penalty_sphere_off.get_active()):
            print("penalty sphere deletion")
            penalty_sphere.remove()
            fig.canvas.draw()
    #
    def distance_measurement_on (self, widget, data=None):
        global cid
        global mid
        global last_mark
        """
        function measurement activated from GUI
        Args:
        - from GUI
        Out:
        - global cid mid which are the event pids for the mouse click and move
        - global last_mark which is the label marking the mouse event activated
        Return:
        - none
        """
        if (self.button_function_measurement_on.get_active()):
            print("measurement on")
#            function_measurement "on" showing first at the ball
            x2, y2, _ = proj_transform(pos_ball_now[0, 0], pos_ball_now[0, 1], pos_ball_now[0, 2], ax1.get_proj())
            last_mark = plt.annotate( "Measurement on",
                    xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            mid = fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)
            cid = fig.canvas.mpl_connect('button_press_event', OnClick)

    def distance_measurement_off(self, widget, data=None):
        """function_measurement = off"""
        if (self.button_function_measurement_off.get_active()):
            print("measurement off")
            last_mark.remove()
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(mid)

    def separate_one_window_on (self, widget, data=None):
        global pOne
        """
        function separate_one_window_on sub-process generated from GUI
        Args:
        - from GUI
        Out:
        - additional 3D window started by a process
        Return:
        - global pOne process id
        """
        if (self.add_separate_one_window_on.get_active()):
            print("open separate additional window 3D")
            pOne = mp.Process(target=OneWindow, args=(s_w_shared, s_d_shared, s_l_shared, el_w_shared, elevation_shared,
                                                       azimut_shared, pos_pb_now_shared, pos_pw_now_shared, pos_ball_now_shared))
            pOne.start()

    def separate_one_window_off(self, widget, data=None):
        """
        function close separate_one_window_on by terminating the sub-process already generated from GUI
        Args:
        - from GUI
        Out:
        - closed additional 3D window
        Return:
        - none
        """
        if (self.add_separate_one_window_off.get_active()):
            print("close separate additional window 3D")
            pOne.terminate()

    def separate_four_window_on (self, widget, data=None):
        global pFour
        """
        function separate_four_window_on sub-process generated from GUI
        Args:
        - from GUI
        Out:
        - additional window 4 views started by a process
        Return:
        - global process id pFour
        """
        if (self.add_separate_four_window_on.get_active()):
            print("open separate window 4x")
            pFour = mp.Process(target=FourWindows, args=(s_w_shared, s_d_shared, s_l_shared, el_w_shared, pos_pb_now_shared,
                                                          pos_pw_now_shared, pos_ball_now_shared))
            pFour.start()

    def separate_four_window_off(self, widget, data=None):
        if (self.add_separate_four_window_off.get_active()):
            print("close separate window 4x")
            pFour.terminate()

    def animation_off_toggled (self, widget, data=None):
        global ani1
        if (self.button_animation_off.get_active()):
            print("pause animation")
            ani1.event_source.stop()
#            fig.canvas.draw()

    def animation_break_toggled(self, widget, data=None):
        global animation_break
        if (self.button_animation_break.get_active()):
            print("break animation; stop animation asap")
            animation_break = True
#
    def animation_on_toggled(self, widget, data=None):
        global ani1
        global animation_slow
        global animation_break
        animation_break = False
        if (self.button_animation_on.get_active()):
            if (self.animation_globalspeedslow.get_active()):
                animation_slow = True
                print("activate animation; slow animation mode according setting")
            else:
                animation_slow = False
                print("activate animation; standard animation mode according setting")
            ani1.event_source.start()
#            fig.canvas.draw()

    def anim_video_on_toggled(self, widget, data=None):  # test the foo status
        global video_page_iter
        global frame_divisor
        global anim_video_on
        print("video button on clicked")
        if (self.button_anim_video_on.get_active()):
            print("video running now")
            video_page_iter = 1
            frame_divisor = int(self.frame_scaling.get_value())
            anim_video_on = True

#            home = os.path.expanduser("~")  # Set the variable home by expanding the user's set home directory
#            if not os.path.exists(os.path.join(home, "Bilder")):  # os.path.join() for making a full path safely
#                os.makedirs(os.path.join(home,"Bilder"))  # If not create the directory, inside their home directory
#
    def anim_video_off_toggled (self, widget, data=None):
        global video_page_iter
        global anim_video_on
        print("video stop button clicked")
        anim_video_on = False
        # verify/confirm the off button is active and create a video only if more than one image exists
        if self.button_anim_video_off.get_active() and video_page_iter > 1:
            print("video stopped and written: overwritting of existing file")
            video_file_name = self.default_filename_video_store.get_text()
            if video_file_name == "":
                video_file_name = "video_name.mp4"
            #
########################################################################################################################
########################################################################################################################
# path of the picture directory to be eventually adapted depending of the computer configuration
#
            os.chdir("/home/family/Bilder")
#
########################################################################################################################
########################################################################################################################

            subprocess.call([
                'ffmpeg', '-y','-framerate', '8', '-i', 'file%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
                video_file_name
                ])  # -y to overwrite  file if it exists
            for file_name in glob.glob("*.png"):
                os.remove(file_name)
            video_page_iter = 0

    def anim_video_pause_toggled (self, widget, data=None):
        global anim_video_on
        print("video button pause clicked")
        if (self.button_anim_video_pause.get_active()):
            print("video confirmed paused")
            anim_video_on = False

    def update_suptitle (self, widget, data=None):
        global plot_suptitle_string
        global plot_suptitle
        plot_suptitle_string = self.suptitle_text.get_text()
        if plot_suptitle_string == "":
            plot_suptitel_string = "underwaterrugby"
        plot_suptitle.remove()
        plot_suptitle = ax1.text2D(0., 1., plot_suptitle_string, fontweight='bold', fontsize=15,
                                   transform=ax1.transAxes,
                                   bbox={'facecolor': 'lightgreen', 'alpha': 0.5, 'pad': 10})
        fig.canvas.draw()


if __name__=="__main__":
    #
    # the additionall windows will be spawned: it makes it quicker
    mp.set_start_method('spawn')
    #
    # swimmingpool size
    s_w = 10.0  # 10m wide
    s_w_shared = mp.Value('f', 10.0)
    s_d = 4.0  # 4m deep
    s_d_shared = mp.Value('f', 4.0)
    s_l = 18.0  # 18m long
    s_l_shared = mp.Value('f', 18.0)
    # exchange lane width
    el_w = 1.0  # normally 3; 1 only to show the side and not 3 because in 3D it makes the view smaller
    el_w_shared = mp.Value('f', 1.0)
    #
    elevation_shared = mp.Value('f', 10.)
    azimut_shared = mp.Value('f', 30.)
    #
    filename_coord_store = ""
    #
    # define/initiate teams blue and white; array
    pos_pb_now = []
    pos_pb_now_shared = mp.Array('f',3*6)
    pos_pb_target = []
    pos_pw_now = []
    pos_pw_now_shared = mp.Array('f',3*6)
    pos_pw_target = []
    pos_pb_deltamove = []
    pos_pw_deltamove = []
    #
    pos_ball_now = []
    pos_ball_now_shared = mp.Array('f',3)
    pos_ball_target = []
    pos_ball_deltamove = []
    #
    clicked_coord = []    # matrix 2x3 for storing coord of clicked points for distance calculation
    clicked_coord.append([0., 0., 0.])
    clicked_coord.append([0., 0., 0.])
    selected_coord = [0., 0., 0.]
    #
    numb_seq = 0 # number of sequences within a loaded position file
    lfd_seq = 1 # default position of the position file is 1
    video_page_iter = 0
    #
    pos_ball_now.append([5.,9.,0.2]) # ball initialized in the middle
    pos_ball_target.append([5.,9.,0.2])
    pos_ball_deltamove.append([0., 0., 0.])
    #
    for i in range(6):
        # distribute the players at the side with the same distance
        # at game start
        pos_pb_now.append([((s_w/6)/2)+i*(s_w/6),1.0, s_d])
        pos_pb_target.append([((s_w/6)/2)+i*(s_w/6),1.0, s_d])
        pos_pw_now.append([s_w - ((s_w / 6) / 2) - i * (s_w / 6), s_l - 1.0, s_d])
        pos_pw_target.append([s_w - ((s_w / 6) / 2) - i * (s_w / 6), s_l - 1.0, s_d])
        pos_pb_deltamove.append([0., 0., 0.])
        pos_pw_deltamove.append([0., 0., 0.])
    #
    # Define numpy array which is faster to work with
    pos_pb_now = np.array(pos_pb_now, dtype='f')
    pos_pb_target = np.array(pos_pb_target, dtype='f')
    pos_pw_now = np.array(pos_pw_now, dtype='f')
    pos_pw_target = np.array(pos_pw_target, dtype='f')
    pos_pb_deltamove = np.array(pos_pb_deltamove, dtype='f')
    pos_pw_deltamove = np.array(pos_pw_deltamove, dtype='f')
    #
    pos_ball_now = np.array(pos_ball_now, dtype='f')
    pos_ball_target = np.array(pos_ball_target, dtype='f')
    pos_ball_deltamove = np.array(pos_ball_deltamove, dtype='f')
    #
    clicked_coord = np.array(clicked_coord, dtype='f')
    selected_coord = np.array(selected_coord, dtype='f')
    #
    player_go_to_clicked = False
    move_running = False
    animation_slow = False   # it means later no pause in the animation function
    anim_video_on = False
    animation_break = False
    frame_divisor = 1
    #
    # initialized the shared data pos in case the "go to" button is clicked after the windows are clicked
    # in order to give a position and not all at zero
    pos_ball_now_shared[0] = pos_ball_now[0, 0]
    pos_ball_now_shared[1] = pos_ball_now[0, 1]
    pos_ball_now_shared[2] = pos_ball_now[0, 2]
    for j in range(6):
        pos_pb_now_shared[j * 3] = pos_pb_now[j, 0]
        pos_pb_now_shared[j * 3 + 1] = pos_pb_now[j, 1]
        pos_pb_now_shared[j * 3 + 2] = pos_pb_now[j, 2]
        pos_pw_now_shared[j * 3] = pos_pw_now[j, 0]
        pos_pw_now_shared[j * 3 + 1] = pos_pw_now[j, 1]
        pos_pw_now_shared[j * 3 + 2] = pos_pw_now[j, 2]
    #
    # create the GUI
    foo=fooclass()
    foo.window_main.show()
    foo.window_main.connect('destroy', gtk.main_quit)
    #
    buffer_label_active_pos = "%03d" % 0
    foo.label_active_pos.set_text(buffer_label_active_pos)

    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    # field
    xG = [0,s_w,s_w,0,0, 0,s_w,s_w,s_w,s_w,s_w, 0, 0,0, 0,s_w]
    yG = [0, 0, 0,0,0,s_l,s_l, 0, 0,s_l,s_l,s_l,s_l,0,s_l,s_l]
    zG = [0, 0, s_d,s_d,0, 0, 0, 0, s_d, s_d, 0, 0, s_d,s_d, s_d, s_d]
    ax1.plot_wireframe (xG,yG,zG,colors= (0,0,1,1))  # blue line game area
    # exchange area
    xW = [s_w,s_w+el_w,s_w+el_w,s_w,s_w,s_w,s_w+el_w,s_w+el_w,s_w+el_w,s_w+el_w,s_w+el_w,s_w,s_w,s_w,s_w,s_w+el_w]
    yW = [0,  0, 0, 0, 0,s_l,s_l, 0, 0,s_l,s_l,s_l,s_l, 0,s_l,s_l]
    zW = [0,  0, s_d, s_d, 0, 0, 0, 0, s_d, s_d, 0, 0, s_d, s_d, s_d, s_d]
    ax1.plot_wireframe (xW,yW,zW,colors= (0,1,1,1))  # light blue line exchange area
    #
    ax1.set_xlabel('Wide')
    ax1.set_ylabel('Length')
    ax1.set_zlabel('Water')
    #
    plot_suptitle_string = "underwaterrugby"
    plot_suptitle = ax1.text2D(0., 1., plot_suptitle_string,fontweight='bold',fontsize=15, transform=ax1.transAxes,
                               bbox={'facecolor': 'lightgreen', 'alpha': 0.5 , 'pad': 10})
    #
    # draw the 2 lines which show the depth
    xG1 = [0, s_w]
    yG1 = [s_d, s_d]
    zG1 = [0, 0]
    ax1.plot_wireframe(xG1, yG1, zG1, colors=(0, 0, 1, 1),linestyle=':')  # blue line
    xG2 = [0, s_w]
    yG2 = [s_l-s_d, s_l-s_d]
    zG2 = [0, 0]
    ax1.plot_wireframe(xG2, yG2, zG2, colors=(0, 0, 1, 1),linestyle=':')  # blue line
    #
    # put the axis fix
    ax1.set_xlim3d(0, s_w+el_w)
    ax1.set_ylim3d(0, s_l)
    ax1.set_zlim3d(0, s_d)
    #
    # use a factor for having y = x in factor
    ax1.set_aspect(aspect=0.15)
    #
    # define the basket1
    draw_basket(ax1, s_w / 2, 0.24, 0., 0.45)
    #
    # define the basket2
    draw_basket(ax1, s_w / 2, s_l - 0.24, 0., 0.45)
    #
    # s=600 is approx 80cm large player at full screen. this parameter can be increased/decreased for
    #  the size appearance of the players
    p_b = ax1.scatter(pos_pb_now[:, 0], pos_pb_now[:, 1], pos_pb_now[:, 2],
                          s=600, alpha = 0.5, c=(0, 0, 1, 1))
    p_w = ax1.scatter(pos_pw_now[:, 0], pos_pw_now[:, 1],
                      pos_pw_now[:, 2], s=600, alpha = 0.5, c="darkgrey")
    p_ball = ax1.scatter(pos_ball_now[:,0], pos_ball_now[:,1],
                      pos_ball_now[:,2], s=100, alpha = 0.5, c="red")
    #
    #Add labels  ..  number and not the numbers defined in the player function (feature tbd)
    for j, xyz_ in enumerate(pos_pb_now):
        annotate3D(ax1, s=str(j+1), xyz=xyz_, fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right',va='bottom')
    for j, xyz_ in enumerate(pos_pw_now):
        annotate3D(ax1, s=str(j+1), xyz=xyz_, fontsize=10, xytext=(-3,3),
                   textcoords='offset points', ha='right', va='bottom')

    Frame = 50  # video will speed up with the Frame divisor from the GUI which
                # will be between 1 and 25 with steps of 5
    # interval=xx in ms between 2 frames; repeat_delay=yy in ms before the animate function is restarted
    # Frame 50    Interval 2   repeat 0  looks like normal appearance. the GUI interaction probably slow everything down
    # the slow modus in the setting will activate a pause timer in the animate function
    Interval_Frame = 2
    Repeat_Delay_Anim = 0
    ani1 = animation.FuncAnimation(fig, animate, frames=Frame, interval=Interval_Frame, blit=False, repeat=True,
                                   repeat_delay=Repeat_Delay_Anim)

    plt.pause(0.001)

    plt.show()
    gtk.main()


