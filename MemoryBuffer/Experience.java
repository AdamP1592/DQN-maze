// Class to hold one experience transition
package MemoryBuffer;
import gamepkg.*;
public class Experience {
    public MazeGame state;//the maze
    public double averageError;
    public Experience(MazeGame state){
        this.state = state;
    }
    public Experience(MazeGame state, double td_errorAvg) {
        this(state);
        this.averageError = td_errorAvg;
    }
}
