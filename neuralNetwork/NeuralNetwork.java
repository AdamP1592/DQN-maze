package neuralNetwork;
import java.util.Arrays;
public class NeuralNetwork implements AutoCloseable {
    NeuralNetworkWrapper nnw = new NeuralNetworkWrapper();
    private int networkStructure[];
    public long nn;
    double[] outputBuffer;
    //to simplify usage in using the neural network
    public NeuralNetwork(int structure[]){
        networkStructure = structure;
        nn = nnw.createNeuralNetwork();
        nnw.setUpNetwork(nn, structure, structure.length);
        outputBuffer = new double[structure[structure.length - 1]];

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                this.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }));
    }
    //private constructor that only gets used by copy, just to prevent
    // the initalization of the structure and to avoid an if statement
    private NeuralNetwork(int structure[], boolean unused){
        networkStructure = structure;
        outputBuffer = new double[structure[structure.length - 1]];
    }
    public double[] forward(double[] inputValues){
        int outputLayerSize = networkStructure[networkStructure.length - 1];
        //pass the values to the wrapper, so the c++ api can handle the network operations
        nnw.forwardPass(nn, inputValues, outputBuffer, inputValues.length, outputLayerSize);

        return outputBuffer;
    }
    public void backProp(double[] expected, int numExpected){
        nnw.backPropagate(nn, expected, numExpected);
    }
    public void backPropRMS(double[] expected, int numExpected){
        nnw.backPropagateRMS(nn, expected, numExpected);
    }
    public NeuralNetwork copy(){
        long newNN = nnw.copyNetwork(nn);

        NeuralNetwork nnCopy = new NeuralNetwork(networkStructure, false);
        nnCopy.nn = newNN;

        return nnCopy;
    }
    //close case so the network is properly destroyed
    @Override
    public void close(){
        nnw.destroyNeuralNetwork(nn);
    }
    
}
