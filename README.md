Underwaterrugby (UWR) 3D simulation and visualization application for training and interactive positioning discussion purpose for player and referee: April29 2018.

It visualize positioning of players in a swimmingpool in 3D (interaction area of player simplified with blue / white spheres; red ball). Furthermore,

It creates images (ending png) for showing positioning of players in the swimmingpool field, inclusive dimension/positions of the swimmingpool, red rugby ball and the baskets at the swimming pool bottom.
It creates positioning files (ending csv) which can be reloaded for dynamic simulation of UWR games. During this running simulation, 3D rotation of the field can be done with the mouse.
It creates video (ending mp4) for showing movement of players in the swimmingpool field. 

The “matplotlib” functionality makes 3D rotating and picture saving possible.

Author pascaldagornet@yahoo.de
Based on underwaterrugby rules available on www.vdst.de and referee training in baden-wuerttemberg by kneer@gmx.net
No warranty: all sport recommendations/rules of www.vdst.de remain valid.

Application tested on 
1. Linux 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux on a Notebook Lenovo T560 screen resolution 1920x1080 GNOME desktop with the latest packages python3 python3-matplotlib gtk+3.0 python3-cairocffi python3-numpy python3-pygobject python3-gi python3-gi-cairo (not exhaustiv).
2. Linux Debian Stretch LXDE9.3 32bit screen 1280x1024 Pentium4 with the packages needed in (1)
3. Linux Debian Stretch Kernel 4.9 32bits 1280x800 Toshiba Satellite Pro

Application NOT tested for now on
1. Windows7 or Windows10 PCs
2. Rasperry PI

Remark: PCs of high spec allow a quicker picture move. If a low spec PC is used, only slow motion will be seen.

Copyright (C) Creative Commons Alike V4.0 https://creativecommons.org/licenses/by-sa/4.0/ 


Installation advice  

The application consist of 2 files
game_uwr.py
game_uwr.glade

The file game_uwr.glade was created with the application “GLADE” for GUI creation in GTK+
It can be placed in any directory; please identify the path: example, /home/uwr_game/. 

The file game_uwr can be placed in any directory; please identify the /pathtothescript/. It will be started from a terminal/console in this path with the command "python3 /pathtothescript/uwr_game.py".

Before the application starts, do following, 

1. create the directory /home/family/Bilder in your PC or adapt the python script game_uwr.py with the new location for storage of images and video. The script area of uwr_game.py to be adapted (edit it and change it with a text editor), are localized between strings of type ”******************”.
2. modify in the uwr_game.py the path of the game_uwr.glade file. The path can be seen below the “class fooclass:” and is between strings of type ”******************”.
3. install all necessary packages in your PC: python3 python3-matplotlib gtk+3.0 python3-cairocffi python3-numpy python3-pygobject python3-gi-cairo (not exhaustiv)

Start the application with "python3 /pathtothescript/uwr_game.py" from a terminal console.


Explanation of startmenue “control”:

see Entries for new player 3D coordinates. 

The value is positive for the  blue team: front positioning, positioning to the right and positioning below (maximum 4m). In order to move the players to the given position, click on "start move" when at "move players" “to menu coord” is selected. 

For the white team: negative value can be given for the white players in case, for example, they are outside the field left in the exchange area observing a penalty.

"move players" Select several defined move of the players via an expanding menue (see screen copy at next page). 

Data from left screen side will be taken into acount only when "move players" “to menu coord” is selected. The move will become effective only by clicking on "start move".
 
Example: choose “penalty against white” in "move players", click "start move", then the players will move according this “penalty against white” positioning: see picture “Start position of a penalty” in the manual. 

The positioning of the ball is controlled independantly from the players positioning and is defined in "move ball", except a moving according a file sequence is choosen (in that case, the ball coordinates of the csv file are automatically overtaken).

"move ball" select where the ball has to be positioned to. 13 different positions can be choosen. 

If an action of type “player” is selectioned, then the fields underneath this entry "move ball", “blue/white” and “1...6” in expanding menue, will be readen in or to move the ball to the desired position.

After clicking "start move", the move of the ball will start.

"coordinates ball" Entries for new ball 3D coordinates. 

The value is positive similar to the coordinates of the blue team: front positioning, positioning to the right and positioning below. In order to move the ball to the given position, click on "start move" when at "move ball" “coordinate” is selected. 

"coord filing" menu for managing coordinates files, containing the 3D position of the ball and players.

“store ok” “to existing file” entry: store the current player and ball position into a file. 
If the file was not choosen in “to existing file”, it will create a new file at each click. The name of the file is indicated in the page “settings”. 
If the file was choosen in “to existing file”, it will add a position at the end of the file at each click (several clicks after each others possible; this will add a new sequence of 13 position lines - 2x 6 players and 1x ball - at each time). 

It is possible to retrieve positions from a csv file and upload them into the application: see the screen copy in the manual (entry “retrieve from”).

A continuous move according all positions of a csv file is possible by choosing in "move players" “acc all seq from file” and then click on "start move" button: if the video button "on" is activated, it will generate a video. Scrolling of the position in the file is possible: for example “to first file pos” “to next file pos” “to last file pos” etc. see scrolling menu in "move players". When this option “acc all seq from file” is choosen, the entries in players coordinates left and "coordinates ball" are ignored. 
The option “till end of file”, if choosen, the entries in screen left (player coodinates) and "coordinates ball" are ignored and the all file sequence will run from the current file position on the screen till last file position. 

"start move" button for generating a move of the players and the ball according the choosen parameter in "move players", for the players, and "move ball", for the ball.

"video" buttons for generating a video; the video generation will be activ when the animation is running/visible in the plot window.

"reset coord menue.." allows to upload the coordinates of the players which can be seen in the game window, into the 3D coordinate menue left of the screen.

Button "update seq pos acc data": when a sequence from a file was uploaded via "coord filing" “retrieve from”, all positions are in the memory. When a position of that file was placed on the screen (via "move players" and choosing for example “first file pos”) and if some changes were made on the coordinates of the players/ball, it is possible to overwrite the sequence in memory with the position on the screen. It helps to modify uploaded sequence files afterward.
Button "store updated seq file": the modified sequence in can be stored again in a file which name is indicated in the menu “settings”.

"animation of the players" "on" or "hold" or "break": when a file of movement sequence (xxx.csv) is uploaded in “retrieve from” and the function is running with “acc file seq”, after clicking on "start move", 
it is possible to “hold” the animation by clicking in this area. This can be usefull in order to pause a video creation or have time to speak to others when an animation is running. 
If “acc file seq” is running, “break” will help to cancel the automatic running of the full sequence in the file (the animation will end at the next file position).
Clicking on “on” will activate again if “hold” was activ. A “on” after a “break” will not do anything.


Explanation of menue “settings”:

"add separate window: additional windows can be opened in case a second desktop is available for presentation to other persons during a rugby discussion/seminar. 
→ A 1x 3D window can be selected: this is the copy of the main window which is already open (measurement annotations and spheres for representation of free and penalty distance will not be showned there).
→ A 4x views windows will be opened/closed.

"video frame divisor"; scalling from 1 to 25: this allow to reduce the number of stored images, which will create the video. The effect will be the video will be much quicker during the movement of players (1 high number of frames making a slow motion video; 25 low number of frame making the video fast or jumping from one position to the other). A number of “2” is creating a video with a normal speed on a high spec PC. 

"default CSV file name" and "default video file name"; entries for giving names to files which can be created/stored: positioning csv file and video file mp4.

"movements of the players and ball in the animation": when the animation of the player is running at the PC screen, a slow motion can be choosen. This has no effect on the video if the recording is running. 

"3D axis positioning" allow a move of the 3D axes according the given values. 
Anyway, the 3D axes can be moved with the mouse by holding the left button and moving the cursor over the window = this is independent of these values in the 3D view. 
It helps to create video and pictures at the same repetitive identified positions.

"update suptitle": the plots have everytime a suptitle (default “underwaterrugby”) and it can be changed. This make possible the commenting of dynamic moving: if the video is “on”, this text/comment will be on the video. 


Explanation of menue “tools”

"free (1/2 sphere)": Activate a half of a sphere at the given position: this represent the 2m distance where any player of the other team should not act during the release of a free. 

"distance meaurement function": a functionality for measuring the distance between 2 players can be activated there. Measurement when “on”: click first a player; click a second player; the distance appear in the command terminal where uwr_game.py was started, at the second click. Further clicking on player will give the distance to the previous player.

"penalty (1/4 sphere)"; a quarter of a sphere can be drawn at the given position: it represent the distance where the goalkeeper should stay as long he is not having full control of the ball during a penalty. He can goes up, outside of that sphere, to take a breath, but not do anything against the player and the ball outside that sphere.

"pos manipulation": when a file of sequences is uploaded, positions (uploaded in memory) can be deleted or added. 
In menue “control”, click the button "store updated file seq” in order to store that modified sequences into a file.


Next (not exhaustiv):

- Test in other linux versions, Microsoft Windows and make necessary improvements of the python script and/or README in order to make the application running out of the box in diverse environment
- create new file positioning sequences and video creation of it 
- create additional pictures of typical game positioning 
- share of generated pictures and video on youtube.com
- Improve/developp functionality: realtime behaviour (time based)
- Move GUI from Gtk3 to Qt5 for use on android tablets


DONT HESITATE TO CONTACT THE AUTHOR FOR RECEIVING VIDEOS AND PICTURES AND FILE SEQUENCES; FEW OF SEQUENCE FILES ARE ALREADY UPLOADED ON GITHUB https://github.com/f4iteightiz/UWR_simulator
  
