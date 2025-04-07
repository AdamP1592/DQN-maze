
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
import java.util.Random;

class dqnMazeGame{
    Json ids = new Json();
    String labelFP = "./data/labels.json";
    int gameSize = 10; 
    int randomSeed = 512;
    int[][] intMap;
    int flattenedSize = gameSize * gameSize;
    int[] networkStructure = {flattenedSize, (int)Math.round(flattenedSize*1), (int)Math.round(flattenedSize * 0.75), (int)Math.round(flattenedSize * 0.75), 64, 64, 32, 32, 16, 8, 4};
    int[] reducedWidthNetwork = {flattenedSize, 96, 64, 32, 32, 32, 16, 16, 8, 8, 4};
    //networks are significantly more stable the less width there is due to rmsbackprop training because there are fewer unused weights

    MazeGame game;
    public void main(){
        setup();        
        
    }
    public void setup(){

        game = new MazeGame(gameSize, gameSize, randomSeed);
        dqn();
        
    }
    public void dqn(){
        //w q e and s outputs

        //layer 1 encodes the data 
        


        duelinDQN();

    }
    public double mapDistanceToValue(double distance, double maxDistance) {
        // Clamp distance to the valid range [0, maxDistance]
        if (distance < 0) {
            distance = 0;
        } else if (distance > maxDistance) {
            distance = maxDistance;
        }
    
        double yMax = 0.1;  // value when distance == 0
        double yMin = -0.1; // value when distance == maxDistance
    
        // Linear interpolation:
        return interpolate(yMin, yMax, distance, maxDistance);
    }

    double interpolate(double min, double max, double value, double maxValue){
        double interpolatedValue = 0;
        if(value > maxValue){
            value = maxValue;
        }
        return min + ((max - min) * (value / maxValue));
    }
    public double calculateEpsilon(int currentStep, int maxSteps, double startingEpsilon) {
        double minEpsilon = 0.2; // Set a minimum epsilon value.
        // Calculate the decay factor as a fraction of remaining steps.
        double decayFactor = 1.0 - ((double) currentStep / maxSteps);
        double epsilon = startingEpsilon * decayFactor;
        // Clamp epsilon to the minimum value.
        if (epsilon < minEpsilon) {
            epsilon = minEpsilon;
        }
        return epsilon;
    }
    double getRewardFromExploring(int pos, int[] movement){
        movement[pos]++;
        //if this is the first time it's been explored
        if(movement[pos] == 1){
            return 0.05;
        }
        return Math.max(-0.1, -0.005 * movement[pos]);
    }
    double interpolateDifficulty(int winCount, int maxWins){
        
        if(winCount > maxWins){
            winCount = maxWins;
        }
        double min = 0.3;

        double max = 1.0;

        return interpolate(min, max, winCount, maxWins);
    }
    public void duelinDQN(){
        networkStructure[networkStructure.length - 1] ++;
        reducedWidthNetwork[reducedWidthNetwork.length - 1] ++;
        
        NeuralNetwork currentMoveModel = new NeuralNetwork(reducedWidthNetwork);
        NeuralNetwork nextMoveModel = currentMoveModel.copy();
        currentMoveModel.close();
        nextMoveModel.close();

        Random r = new Random();
        int numEps = 8000;
        int numWinsUntilMax = 400;//should be 500
        int numActions = 4;
        int numTests = 1;

        
        boolean backupLoaded = false;
        boolean backupUpdated = false;

        int resetCount = 0;
        int winCount = 0;
        int backupUsedCount = 0;

        double[] rewards = new double[numEps];
        int[] winCounts = new int[numTests];
        double maxPossibleDistance = Math.sqrt(Math.pow(gameSize, 2)+ Math.pow(gameSize, 2));
                

        for(int t = 0; t < numTests; t++){
            currentMoveModel = new NeuralNetwork(reducedWidthNetwork);
            nextMoveModel = currentMoveModel.copy();
            NeuralNetwork backupMoveModel = currentMoveModel.copy();
            double maxDifficulty = 0.0;

            double dDistanceReward = -0.2 / maxPossibleDistance;
            double distanceRewardCap = 0.1;
            double distanceModifier = distanceRewardCap/dDistanceReward;

            double decayExploration = 1.0;

            for(int episode = 0; episode < numEps; episode++){
                decayExploration = interpolate(1.0, 0.5, winCounts[t], numWinsUntilMax);
                // checking for network explosion as a result of a hyperaggressive gradient in
                // rmsbackprop tanking the network Need to update nn framework with gradient normalization
                // and batch training to ease noise out and allow network to adjust, maybe do an bell-shaped curve
                // for learning rate to ease the network into the new training data
                double[] outputs = currentMoveModel.forward(processMaze(mazeToInt()));
                for(int i = 0; i < outputs.length; i++){
                    if(Double.isNaN(outputs[i]) ){
                        
                        currentMoveModel.close();
                        nextMoveModel.close();
                        if(!backupUpdated && backupLoaded){
                            System.out.println("loading from backup");
                            currentMoveModel = backupMoveModel.copy();
                            backupUsedCount++;
                        }
                        else{
                            System.out.println("Resetting network");
                            currentMoveModel = new NeuralNetwork(reducedWidthNetwork);
                            resetCount++;
                        }
                        nextMoveModel = currentMoveModel.copy();
                        //load from backup if the network explodes. Would prefer to have an additional option where if you have to load a backup

                        backupLoaded = true;
                        backupUpdated = false;
                        System.out.flush();
                        System.out.println("\n\n\n\n");
                        break;
                    }
                }

                int randNumber = r.nextInt();

                //every 2 wins increase difficulty 
                double difficulty = interpolateDifficulty((int)Math.floor(winCounts[t]), numWinsUntilMax);
                System.out.println("Difficulty: " + difficulty);
                //on the generator, every path in the maze is stored in a list, sorted, and then the % difficulty is relative to the furthest path away from the randomly placed player.
                //instead of this you could just set the path in the bfs by passing the running average length of the furthest path away from the maze. 
                //as the max value and as you do the bfs once you pass the difficulty % * that average max length to get the length of the path, and in the bfs, just stop once you
                //reach a distance of that value
                game = new MazeGame(gameSize, gameSize, randNumber, difficulty);
                int[][] mazeMap = mazeToInt();

                double[] processedMaze = processMaze(mazeMap);
                int[] visitedAreas = new int[gameSize * gameSize];
                double maxEpsilon = 0.9;
                double gamma = 0.99;
                double distanceGamma = 0.9;

                boolean endCondition = false;
                //q, w, e, s
                
                int step = 0;
                int maxSteps = 100;

                int[] movementBuffer = new int[maxSteps];
                double epsilon = maxEpsilon;
                double[] gradientBuffer = new double[numActions + 1];
                int epWinCount = 0;
                while(!endCondition){
                    if(episode == numEps - 1){
                        printMaze();
                    }
                    int currentHealth = game.getHealth();
                    int action = 0;

                    //    CALCULATE Q VALUES USING STATE VARIABLE

                    //basic setup for any dqn
                    double[] qValues = new double[numActions];
                    double[] networkOutput = currentMoveModel.forward(processedMaze);
                    double v = networkOutput[numActions];


                    // calc qValues with action value
                    double avgAdvantage = 0;
                    for(int i = 0; i < numActions; i++){
                        avgAdvantage += networkOutput[i];
                    }
                    avgAdvantage/= numActions;
                    double maxQ = v + (networkOutput[0] - avgAdvantage);//some value thats significantly less than any possible output
                    
                    action = 0;
                    for(int i = 1; i < numActions; i++){
                        qValues[i] = v + (networkOutput[i] - avgAdvantage);
                    }
                    if(Math.random() < epsilon){
                        action = (int) (Math.random() * numActions);
                        maxQ = qValues[action];
                        //num actions = 4, the 4th index is the v value
                    }
                    else{
                        for(int i = 0; i < numActions; i++){
                            if(qValues[i] > maxQ){
                                action = i;
                                maxQ = qValues[i];
                            }
                        }
                    }

                    double preMoveDistance = game.distanceToGoal();
                    //Reward based on game logic(if the alogrithm tried to run into the wall there's a strong negative punishment)
                    double runIntoWallPunishment = game.move(action);
                    
                    double postMoveDistance = game.distanceToGoal();

                    //every time you move you get a negative reward
                    double reward = -0.1; //moderate punishment for each step

                    int newHealth = game.getHealth();
                    //punishment rules

                    reward += runIntoWallPunishment;

                    if(preMoveDistance == postMoveDistance){
                        reward -= 0.08; //mild punishment for turning
                    }
                    else{
                        //reward exploration while punish revisiting other tiles
                        int[] playerPos = game.getPlayerPosition();
                        int coordsToPos = (playerPos[1] * gameSize) + playerPos[0];
                        //reward += decayExploration * getRewardFromExploring(coordsToPos, visitedAreas);
                    }

                    //reward moving towards goal, punsh moving away from goal. (dMoveValue = sqrt((x - goalX)^2 + (y-goalY^2)) - sqrt((newX - goalX)^2 + (newY - goalY)));
                    // since a move can only happen in one direction, for the x direction it's sqrt((x-goalX)^2 + goalY^2) - sqrt((newX - goalX)^2 - goalY)
                    double preMoveValue = mapDistanceToValue(preMoveDistance, maxPossibleDistance);
                    double postMoveValue = mapDistanceToValue(postMoveDistance, maxPossibleDistance);

                    //decay move reward and punishment as the difficulty gets higher, because the further you are placed from the goal
                    //the more likely you are to need to move away from the goal to move around obstacles to get to the goal
                    double moveReward = (preMoveValue - postMoveValue ) * distanceModifier * decayExploration;
                    //System.out.println(moveReward);
                    if(moveReward != 0.00){
                        //moveReward = (Math.abs(moveReward)/moveReward) * Math.min(Math.abs(moveReward), distanceRewardCap);
                    }

                    //System.out.println("move reward " + moveReward);
    
                    double newDistanceReward = distanceGamma * (1 - (postMoveDistance / maxPossibleDistance)) - (1 - (preMoveDistance / maxPossibleDistance));
                    reward += moveReward;
                    
                    //conclusion rules
                    if(game.isGameOver()){
                        if(game.isDead()){
                            reward = -1;
                        }
                        if(game.goalReached()){
                            reward = 1;
                            winCount++;
                            epWinCount++;
                            winCounts[t]++;
                            System.out.println("Game won");

                        }
                        endCondition = true;
                    }   
                    if(!game.goalReached() && step + 1 >=maxSteps){
                        reward = -1;
                    }
                    rewards[episode] += reward;
                    double[] nextState = processMaze(mazeToInt());
                    double[] qNext = nextMoveModel.forward(nextState);


                    //gets the next move
                    double maxNextQ = 0.0;
                    if(!endCondition){
                        maxNextQ = qNext[0];
                        for(int i = 1; i < qNext.length; i++){
                            maxNextQ = Math.max(maxNextQ, qNext[i]);
                        }
                    }

                    double endConModifier = (endCondition ? 0.0: gamma * maxNextQ);
                    
                    //System.out.println("distance reward: " + newDistanceReward);
                    //displays reward for the last 25% of episodes
                    if(episode > numEps*0.99){
                        //System.out.println("Reward" + reward + " endConModifier" + gamma + maxNextQ);
                    }
                    //catch case if there is a gameover without dying or winning

                    //generates target array using a trick where you calculate the error on this side
                    //the pass the error + the output of the network
                    //since the error in the network is just o - e you make the
                    //passing o with the error yields o - o + error
                    double targetQ = reward + endConModifier;
                    double[] expected = new double[numActions + 1];
                    double TD_error = (qValues[action] - targetQ);
                    
                    double[] actionGradients = new double[numActions + 1];
                    //fill array as if all values aren't chosen values
                    for(int i = 0; i < numActions; i++){
                        //should be:
                        actionGradients[i] = (TD_error/(double) numActions);
                        //but that generates a lot of noise and makes the network less stable. This is only an issue
                        //because I dont normalize gradients and dont batch train.
                        //expected[i] = 0;
                    }
                    //rewrite action and state values with custom error
                    actionGradients[action] = - (TD_error * (1.0 - (1.0 / (double) numActions)));
                    actionGradients[numActions] = - TD_error;

                    
                    double[] expectedValues = new double[numActions + 1];
                    for(int i = 0; i < numActions; i++){
                        gradientBuffer[i] += actionGradients[i];
                        expected[i] = qValues[i] - gradientBuffer[i];
                    }
                    
                    expected[action] = qValues[action] + gradientBuffer[action];
                    expected[numActions] = networkOutput[numActions] - gradientBuffer[numActions];

                    if(step % 100 == 0){// every 100 steps backprop
                        currentMoveModel.backPropRMS(expectedValues, expectedValues.length);
                        for(int i = 0; i < gradientBuffer.length; i ++){
                            gradientBuffer[i] = 0.0;
                        }
                    }
                    //passes target array to the current move model
                    //currentMoveModel.backPropRMS(expected, expected.length);

                    //adds action to the movement buffer
                    movementBuffer[step] = action;
                    step++;

                    //sets end condition
                    if(step >= maxSteps || game.isGameOver() || game.isDead()){
                        endCondition = true;

                    }

                    //random choice decay scehdule
                    int stepTotal = step + (episode * maxSteps);
                    int maxTotalSteps =  maxSteps * numEps;
                    epsilon = calculateEpsilon(stepTotal, maxTotalSteps, maxEpsilon);

                    //copys the current move network to the next move network
                    if(stepTotal % 400 == 0){
                        System.out.println("Copying network");
                        nextMoveModel.close();
                        nextMoveModel = currentMoveModel.copy();
                    }
                    if(episode > numEps - 100){
                        GameWindow w = new GameWindow();
                        w.cols = gameSize;
                        w.rows = gameSize;
                        w.seed = randNumber;
                        w.game = game;
                        w.startWindow();
                    }
                }
                outputs = currentMoveModel.forward(processMaze(mazeToInt()));
                if(gradientBuffer[0] != 0.0){// if there are any left over gradients train on those
                    double[] expected = new double[numActions + 1];
                    for(int i = 0; i < gradientBuffer.length; i++){
                        gradientBuffer[i]/=step;
                        expected[i] = outputs[i] + gradientBuffer[i];
                    }
                }
                if(episode % 40 == 0){//every 40 epsides copy the move model over to the backup
                    backupMoveModel.close();
                    backupMoveModel = currentMoveModel.copy();
                    backupUpdated = true;
                    backupLoaded = false;
                }
                //prints average reward for the episode
                rewards[episode] /= step;
                System.out.println(rewards[episode]);
                System.out.println("Episode end: " + episode);
                System.out.println("EpisodeWinCount: " + epWinCount);
                System.out.println("Win count " + winCounts[t]);
                
            }
            
            System.out.println("Training end");
            System.out.println(winCount);
            
        }
        System.out.println("Reset Count: "+ resetCount);
        System.out.println("Backups Used: "+ backupUsedCount);
        for(int i = 0; i < numTests; i++){
            System.out.println(winCounts[i]);
        }
        currentMoveModel.close();
        nextMoveModel.close();
        //prints 1 in 20 episode reward average
    
    }
    private void printMaze(){
        for(int[] row: intMap){
            for(int pos: row){
                System.out.print(pos + " ");
            }
            System.out.println();
        }
        System.out.println();
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


        intMap = new int[0][0];
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
        /*

        */
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