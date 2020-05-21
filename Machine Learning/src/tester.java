import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class tester {
	public static void main(String[] args) {
		Neural_Network net = new Neural_Network(new int[] {2, 10, 2});
		System.out.println("Before test");
		System.out.println(Arrays.toString(net.predict(new double[] {1, 1})));
		System.out.println(Arrays.toString(net.predict(new double[] {1, 1})));
		System.out.println(Arrays.toString(net.predict(new double[] {0, 0})));
		System.out.println(Arrays.toString(net.predict(new double[] {1, 0})));
		List<Pair<double[], double[]>> list = new ArrayList<>();
		list.add(new Pair<double[], double[]>(new double[] {1, 0}, new double[] {1 , 2}));
		list.add(new Pair<double[], double[]>(new double[] {1, 1}, new double[] {1, 3}));
		list.add(new Pair<double[], double[]>(new double[] {1}, new double[] {1, 3}));
		list.add(new Pair<double[], double[]>(new double[] {0, 0}, new double[] {3, 0}));
		net.train(15, 0, list);
		System.out.println("after test");
		System.out.println(Arrays.toString(net.predict(new double[] {0, 1})));
		System.out.println(Arrays.toString(net.predict(new double[] {1, 1})));
		System.out.println(Arrays.toString(net.predict(new double[] {1, 0})));
		System.out.println(Arrays.toString(net.predict(new double[] {0, 0})));
	}
}
