
import gamepkg.*;
import neuralNetwork.NeuralNetwork;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import java.lang.StringBuilder;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.io.BufferedWriter;
import java.io.FileWriter;

import java.io.BufferedReader;
import java.io.FileReader;

import java.io.IOException;

class dqnMazeGame{
    Json ids = new Json();
    String labelFP = "./data/labels.json";
    int gameSize = 25; 
    int randomSeed = 512;

    MazeGame game;
    public void main(){
        setup();        
        
    }
    public void setup(){

        game = new MazeGame(gameSize, gameSize, randomSeed);

        int[][] mazeMap = mazeToInt();
        double[] processedMaze = processMaze(mazeMap);
        
        int flattenedSize = processedMaze.length;

        //w q e and s outputs
        int[] networkStructure = {flattenedSize, 128, 64, 32, 8, 4} ;
        
    }
    //normalizes and flattens maze
    public double[] processMaze(int[][] mazeMap){
        int [] flattenedMaze = new int[0];
        double [] normalizedFlattenedMaze = new double[0];
        if(mazeMap.length == 0){
            System.err.println("Error maze empty");
            return normalizedFlattenedMaze;
        }
        if(mazeMap[0].length == 0){
            System.err.println("Error maze not filled");
            return normalizedFlattenedMaze;
        }
        flattenedMaze = new int[mazeMap.length * mazeMap[0].length];
        normalizedFlattenedMaze = new double[mazeMap.length * mazeMap[0].length];


        int maxValue = 0;
        int minValue = 100;
        for(int i = 0; i < mazeMap.length; i++){
            for(int j = 0; j < mazeMap[i].length; j++){

                int value = mazeMap[i][j];
                int flattenedIndex = (i * mazeMap[i].length) + j;
                flattenedMaze[flattenedIndex] = value;
                maxValue = Math.max(maxValue, value);
                minValue = Math.min(minValue, value);
            }
        }
        normalizedFlattenedMaze = normalizeArr(flattenedMaze, minValue, maxValue);
        return normalizedFlattenedMaze;
    }
    private double[] normalizeArr(int[] arr, int minVal, int maxVal){
        double[] normalizedArr = new double[arr.length];

        for(int i = 0; i < arr.length; i++){
            int value = arr[i];
            double normalizedValue = 0.0;
            if(minVal != maxVal){
                normalizedValue = (double)(value - minVal)/(maxVal - minVal);
            }
             
            normalizedArr[i] = normalizedValue;
        }

        return normalizedArr;
    }
    public void freeNetwork(NeuralNetwork nn){
        nn.close();
    }
    public NeuralNetwork createNN(int strucutre[]){
        NeuralNetwork nn = new NeuralNetwork(strucutre);
        return nn;
    }

    public Entity[][] createMaze(int mazeSeed){
        int width = 25;
        int height = 25;
        mazeGenerator m = new mazeGenerator(width, height, mazeSeed);
        Entity[][] entityMap = m.getMaze();
        
        return entityMap;

    }
    public int[][] mazeToInt(){

        Entity[][] mazeMap = game.maze;

        int[] playerPos = game.getPlayerPosition();


        int[][] intMap = new int[0][0];
        //catch case if maze is empty
        if(mazeMap.length == 0){
            return intMap;
        }
        if(mazeMap[0].length == 0){
            return intMap;
        }

        intMap = new int[mazeMap.length][mazeMap[0].length];
        ids = retrieveLabels();
        int lastInt = -1;
        if(ids.map.size() != 0){
            lastInt = Integer.parseInt(ids.getLastValue());
        }
        

        for(int i = 0; i < mazeMap.length; i++){
            for(int j = 0; j < mazeMap[i].length; j++){
                String tileClassString = mazeMap[i][j].getClass().getSimpleName();

                //if tile has an effect, change the class name out with the effect
                if(tileClassString.equals("TrappedFloor")){
                    tileClassString = mazeMap[i][j].getEffect().getClass().getSimpleName();
                }
                if(i == playerPos[1] && j == playerPos[0]){
                    tileClassString = "Player";
                }
                String id = ids.get(tileClassString);


                if(id == null){
                    lastInt++;
                    String lastValue = String.valueOf(lastInt);
                    ids.put(tileClassString, lastValue);
                    id = lastValue;
                }   

                intMap[i][j] = Integer.parseInt(id);
            }
        }
        for(int[] row: intMap){
            for(int pos: row){
                System.out.print(pos + " ");
            }
            System.out.println();
        }
        storeLabels();
        return intMap;
    }
    public void storeLabels(){
        try(BufferedWriter w = new BufferedWriter(new FileWriter(labelFP))){
            w.write(ids.getJsonString());
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
    public Json retrieveLabels(){
        Json j = new Json();
        StringBuilder content = new StringBuilder();

        try(BufferedReader r = new BufferedReader(new FileReader(labelFP))){
            String line;
            while ((line = r.readLine()) != null) {
                content.append(line).append("\n");
            }
            j.convertJsonToMap(content.toString());
        }   
        catch(IOException e){
            e.printStackTrace();
        }
        return j;
    }

}
class Json{
    public Map<String, String> map;

    public Json(){
        map = new LinkedHashMap<>();
    }
    public void put(String key, String value){
        map.put(key, value);
    }
    public String get(String key){
        return map.get(key);
    }
    public String getLastValue(){
        String lastVal = "";
        for(String key: map.keySet()){
            lastVal = map.get(key);
        }

        return lastVal;
    }
    public String getJsonString(){
        StringBuilder sb = new StringBuilder();
        Set<String> keys = map.keySet();

        if(keys.isEmpty()){
            return "";
        }

        sb.append("{");
        for(String key: keys){
            String jsonEntry = String.format("\"%s\": \"%s\",\n", key, map.get(key));
            sb.append(jsonEntry);
        }
        //remove last comma and newline
        sb.setLength(sb.length() - 2);
        //close bracket
        sb.append("}");

        return sb.toString();
    }
    public void convertJsonToMap(String jsonString){
        //to prevent overloading the ram by splitting the whole string

        String[] objects = jsonString.split(",\\s*\\n");

        if (objects.length > 0) {
            objects[0] = objects[0].replaceFirst("^\\{", ""); // Remove leading '{'
            int last = objects.length - 1;
            objects[last] = objects[last].replaceFirst("\\}$", ""); // Remove trailing '}'
        }

        Pattern p = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"");
        
        for (String obj : objects) {
            Matcher m = p.matcher(obj);
            while (m.find()) {
                String key = m.group(1);
                String value = m.group(2);
                map.put(key, value);
            }
        }
    }
    
}