# Screen Recorder

Screen recorder provides processing with OpenCV so you can capture screenshots, watch regions of your screen, and detect faces with either CLI or a GUI via [Gooey](https://github.com/chriskiehl/Gooey).

## Help

```
usage: . [-h] [-n FILE_NAME] [-b BBOX BBOX BBOX BBOX] [-w] [--stats]
         [-m {rgb,edges,faces,blue,red,green,rank,median,laplacian}]

Video or screenshotting tool

optional arguments:
  -h, --help            show this help message and exit
  -n FILE_NAME          set a file name
  -b BBOX BBOX BBOX BBOX
                        Define a bounding box
  -w                    Record and show part of the screen
  --stats               show stats
  -m {rgb,edges,faces,blue,red,green,rank,median,laplacian}
                        Video mode
```
