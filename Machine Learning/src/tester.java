import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class tester {
	public static void main(String[] args) {
		String LABEL_FILE = "src\\" + "trainLabel.idx1-ubyte";
		String IMAGE_FILE = "src\\" + "trainImage.idx3-ubyte";
		Neural_Network net = new Neural_Network(new int[] { (28 * 28), 200, 80, 10 });
		List<Pair<double[], double[]>> data = MNIST.combine(MNIST.getImages(IMAGE_FILE), MNIST.getLabels(LABEL_FILE));
		Collections.shuffle(data);
		net.train(3, 100, data, 0.01);
		int test = 10001;
		System.out.println("TRIAL:");
		System.out.println("X:");
		System.out.println(MNIST.render(data.get(test).getA()));
		System.out.println("Y:");
		System.out.println(Arrays.toString(data.get(test).getB()));
		System.out.println("Prediction: " + Neural_Network.argMax(net.predict(data.get(test).getA())));
	}
}
