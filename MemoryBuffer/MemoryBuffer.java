package MemoryBuffer;
import java.util.Random;

public class MemoryBuffer {
    private Experience[] buffer;
    private int capacity;
    private int size = 0;
    private int index = 0;
    private Random rand = new Random();

    public MemoryBuffer(int capacity) {
        this.capacity = capacity;
        this.buffer = new Experience[capacity];
    }

    public void add(Experience exp) {
        buffer[index] = exp;
        index = (index + 1) % capacity;
        if (size < capacity) {
            size++;
        }
    }

    public Experience[] sample(int batchSize) {
        Experience[] batch = new Experience[batchSize];
        for (int i = 0; i < batchSize; i++) {
            int idx = rand.nextInt(size);
            batch[i] = buffer[idx];
        }
        return batch;
    }

    public int size() {
        return size;
    }
}
