package neuralNetwork;
public class NeuralNetwork implements AutoCloseable {
    NeuralNetworkWrapper nnw = new NeuralNetworkWrapper();
    private int networkStructure[];
    long nn;
    //to simplify usage in using the neural network
    public NeuralNetwork(int structure[]){
        networkStructure = structure;
        nn = nnw.createNeuralNetwork();
        nnw.setUpNetwork(nn, structure, structure.length);
    }
    public double[] forward(double[] inputValues){
        int outputLayerSize = networkStructure[networkStructure.length - 1];
        double[] outputBuffer = new double[outputLayerSize];

        //pass the values to the wrapper, so the c++ api can handle the network operations
        nnw.forwardPass(nn, inputValues, outputBuffer, inputValues.length, outputLayerSize);

        return outputBuffer;
    }
    public void backProp(long nn, double[] expected, int numExpected){
        nnw.backPropagate(nn, expected, numExpected);
    }
    public void backPropRMS(long nn, double[] expected, int numExpected){
        nnw.backPropagateRMS(nn, expected, numExpected);
    }
    public long copy(long nn){
        long newNN = nnw.copyNetwork(nn);
        return newNN;
    }
    //close case so the network is properly destroyed
    @Override
    public void close(){
        nnw.destroyNeuralNetwork(nn);
    }
    
}
