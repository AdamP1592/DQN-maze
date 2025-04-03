package neuralNetwork;
import java.io.File;

public class NeuralNetworkWrapper {
    static {
        //so you dont have to run java.library.path
        String relativePath = "neuralNetwork/nn/NeuralNetworkLib.dll";
        String absPath = new File(relativePath).getAbsolutePath();
        System.out.println(absPath);
        System.load(absPath);// Make sure NeuralNetworkLib.dll is in the java.library.path
    }
    
    // Native method declarations.
    public native long createNeuralNetwork();
    public native long copyNetwork(long nn);
    public native void destroyNeuralNetwork(long nn);
    public native void setUpNetwork(long nn, int[] structure, int length);
    public native void forwardPass(long nn, double[] inputValues, double[] outputBuffer, int numInputs, int numOutputs);
    public native void backPropagateRMS(long nn, double[] expected, int numExpected);
    public native void backPropagate(long nn, double[] expected, int numExpected);
    
    // Example usage:
    /*
        public static void main(String[] args) {

        System.out.println("Starting");
        NeuralNetworkWrapper nnw = new NeuralNetworkWrapper();

        System.out.println("nnw initialized");
        // Create and set up the neural network.
        long nnHandle = nnw.createNeuralNetwork();
        long copyHandle;

        System.out.println("network created");
        int[] structure = {5, 10, 2};  // 5 inputs, 10 hidden, 2 outputs
        nnw.setUpNetwork(nnHandle, structure, structure.length);

        System.out.println("Network setup");
        
        // Prepare input data (5 inputs) and preallocate output buffer (2 outputs).
        double[] inputs = {0.5, 0.75, 0.25, 0.1, 0.3};
        double[] outputs = new double[structure[structure.length - 1]];
        double[] expected = {0.5, 0};

        System.out.println(outputs.length);
        // Perform a forward pass on the base network
        System.out.println("Starting forwardPass");

        nnw.forwardPass(nnHandle, inputs, outputs, inputs.length, outputs.length);
        System.out.println("First Network Outputs");
        for (double out : outputs) {
            System.out.println(out);
        }
        copyHandle = nnw.copyNetwork(nnHandle);
        // forwardPass the copy to see the copies outputs
        System.out.println("Copy Network Outputs");
        nnw.forwardPass(copyHandle, inputs, outputs, inputs.length, outputs.length);
        for (double out : outputs) {
            System.out.println(out);
        }
        System.out.println("Back prop base network");
        //back prop the base network
        nnw.backPropagateRMS(nnHandle, expected, expected.length);
        
        //forward pass each network to confirm different outputs
        System.out.println("First Network Outputs");
        nnw.forwardPass(nnHandle, inputs, outputs, inputs.length, outputs.length);
        for (double out : outputs) {
            System.out.println(out);
        }
        System.out.println("Copy Network Outputs");
        nnw.forwardPass(copyHandle, inputs, outputs, inputs.length, outputs.length);
        for (double out : outputs) {
            System.out.println(out);
        }
        
        // Example expected outputs for training.
        
        
        // Clean up.
        nnw.destroyNeuralNetwork(nnHandle);
        nnw.destroyNeuralNetwork(copyHandle);
    }
    */
}
