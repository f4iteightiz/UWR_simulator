# UWR_simulator
#################################################################################################################################
Underwaterrugby 3D simulation and visualization application for training discussion purpose for players and referees.

It visualize positioning of players in a swimmingpool in 3D (interaction area of player simplified with a sphere).

It creates positioning files which can be reloaded for dynamic simulation of UWR games.

It creates video for showing movement of players in the swimmingpool field.

The “matplotlib” functionality can be used: 3D rotating, picture saving.
Make sure the directory /home/family/Bilder exists in your PC or adapt the python script game_uwr.py with the new location for storage of video images.
Put the GUI file into /home/family/glade/game_uwr_180401.glade and/or adapt the name and/or file location in the script game_uwr.py
Start with "python3 uwr_game.py" from the console.
Author pascaldagornet@yahoo.de
Based on underwaterrugby rules available on www.vdst.de and referee training in baden-wuerttemberg by kneer@gmx.net
No warranty: all sport recommendations/rules of www.vdst.de remain valid.

Application tested on 

1. Linux 4.9.0-6-amd64 #1 SMP Debian 4.9.82-1+deb9u3 (2018-03-02) x86_64 GNU/Linux on a Notebook Lenovo T560 screen resolution 1920x1080 GNOME desktop with the latest packages python3 python3-matplotlib gtk+3.0 python3-cairocffi python3-numpy python3-pygobject python3-gi python3-gi-cairo (not exhaustiv).
2. Linux Debian Stretch LXDE9.3 32bit screen 1280x1024 Pentium4 with the packages needed in (1)
3. Linux Debian Stretch Kernel 4.9 32bits 1280x800 Toshiba Satellite Pro


Explanation of page "control”:

Left side player blue 1..6 and white 1..6 entries: Entries for new player 3D coordinates. The value is positive for blue: at each side, front positioning, positioning to the right and positioning below (maximum 4m). In order to move the players to the given position, click on button "start move" when at "move players" “to menu coord” is selected. Negative value can be given for the white players in case they are outside the field left (in the exchange area observing a penalty for example).

"move players" select several defined move of the players. Data from the players1..6 blue/white entries will be taken into acount only when “to menu coord” is selected. The move will become effective only by clicking on "start move". 
Example: choose “penalty against white”, click "start move", then the players will move according this “penalty against white” positioning. The positioning of the ball is controlled separately from the players positioning: see "move ball" entries.

"move ball"  select where the ball has to be positioned to. If a “player” is selectioned, then the fields below, “blue/white” and “1...6”, will be readen in or to move the ball to the desired position after clicking "start move".

"coordinates" allow a move of the 3D axes according the given elevation & azimut values (anyway, it can be moved interactively independent of these values in the 3D view). It helps to create video and pictures at similar repetitive identified positions.

"coord filing" store the current player and ball position into a file. If the file is not given, it will create a new file at each click. It is possible to retrieve positions from a file and upload them into the application. A continuous move according that selected positions is possible by choosing in "move players" “acc all seq from file” and then click on "start move": if the video button of "anim+video" is activated, it will generate a video. Other scrolling across the position file (see scrolling menu in "move players") are possible. When this option “acc all seq from file”, the entries in the left (all coodinates of players 1..6 and blue/white) and "move ball" are ignored. 

Format of the file; it has the endig .csv and can be uploaded into Libroffice calc; this is a repetitive sequence of 13 lines of 3 numbers separated by a “,”. The line 1 to 6 are the 3D coordinates X Y Z of the player blue 1 to 6. The lines 7 to 12 are the 3D coordinates of the players white 1 to 6. The line 13 is the 3D coordinate of the ball. Example:

0.83,9,4 (= player1 X Y Z)
2.5,9,4
4.17,9,4
5.83,9,4
7.5,9,4
9.17,9,4
9.17,17,4
7.5,17,4
5.83,17,4
4.17,17,4
2.5,17,4
0.83,17,4
4.17,9.6,4
… (next 13 lines)

"move players" button for generating a move of the players and the ball according the choosen parameter in "move players" and "move ball".

"anim+video"  buttons for generating a video.

"reset coord menue.." it allow to upload the coordinates of the players which can be seen in the game window into the 3D coordinate menue on the left side. The coordinates in the left side of the menue dont move automatically according the moves defined by "move players".

"speed" entries: for time based movements (depending of speed) not implemented yet. 


Explanation of page "settings”:

"add separate window" area: additional windows can be opened in case a second desktop is available for presentation to other persons during a rugby discussion/seminar. 
→ A 1x 3D window can be selected: this is the copy of the main window which is already open (measurement annotations and spheres for representation of free and penalty distance will not be showned there).
→ A 4x views windows will be opened/closed 

"frame scalling video.." scalling from 1 to 20: this allow to reduce the number of stored images creating later the video. The effect will be the video will be much quicker during the movement of players (1 high speed; 20 low speed).

"default file name.." entries for giving names to files which can be created/stored.


Explanation of page "tools”:

"free (1/2 sphere)" Activate a half of a sphere at the given position: this represent the 2m distance where any player of the other team should not act during the release of a free. 

"distance measurement function" a functionality for interactive measuring the distance between 2 players can be activated there. Measurement when “on”: click first a player; click a second player; the distance appear in the command window, where uwr_game.py was started, at the seconmd click. Further clicking on player will give the distance to the previous player.

"penalty (1/4 sphere)" a quarter of a sphere can be drawn at the given position: it represent the distance where the goalkeeper should stay as long he is not having full control of the ball during a penalty. He can goes up, outside of that sphere, to take a breath, but not do anything against the player and the ball outside that sphere.


Next to come (not exhaustiv):

- Test in other linux versions, LXDE, screen size, Microsoft Windows and make necessary improvements of the python script and/or README in order to make the application running out of the box in diverse environment
- create new file positioning sequences and video creation of it 
- create pictures of typical game positioning 
- share of generated pictures and video on youtube.com
- edit/change file of movement sequences
- Improve/developp functionality: realtime behaviour (time based)
- create pictures and file sequences and video to share ou youtube.com
- Move GUI from Gtk3 to Qt5
