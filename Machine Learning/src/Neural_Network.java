import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Neural_Network {
	private double[][][] weights;
	private double[][] bias;
	private double[][] z;
	private int[] in;

	interface Predict<T> {
		T func(double[] input, int layer);
	}

	interface Backprop<T> {
		T func(double[][][] tempWeights, double[][] tempBias, double[] derivatives, double[] input, int layer);
	}

	public Neural_Network(int[] in) {
		this.in = in;
		weights = init(in, new double[in.length - 1][][]);
		bias = init(in, new double[in.length - 1][]);
		z = init(in, new double[in.length - 1][]);

	}

	private double[][][] init(int[] in, double[][][] x) {
		Random r = new Random();
		for (int i = 0; i < x.length; i++) {
			x[i] = new double[in[i]][in[i + 1]];
			for (int j = 0; j < x[i].length; j++) {
				for (int u = 0; u < x[i][j].length; u++) {
					x[i][j][u] = r.nextDouble();
				}
			}
		}
		return x;
	}

	private double[][] init(int[] in, double[][] x) {
		for (int i = 0; i < x.length; i++) {
			x[i] = new double[in[i + 1]];
			for (int j = 0; j < x[i].length; j++) {
				x[i][j] = 0;
			}
		}
		return x;
	}

	final Predict<double[]> predict = (input, layer) -> {
		if (layer > weights.length - 1) {
			return input;
		} else {
			double[] t = new double[bias[layer].length];
			for (int i = 0; i < bias[layer].length; i++) {
				double sum = 0;
				for (int x = 0; x < weights[layer].length; x++) {
					sum += weights[layer][x][i] * input[x];
				}
				sum += bias[layer][i];
				z[layer][i] = sum;
				t[i] = sigmoid(sum);
			}
			return this.predict.func(t, ++layer);
		}
	};

	final Backprop<Pair<double[][][], double[][]>> backprop = (tWeights, tBias, derivatives, input, layer) -> {
		if (layer < 0) {
			return new Pair<double[][][], double[][]>(tWeights, tBias);
		} else if (layer == 0) {
			for (int i = 0; i < weights[layer][0].length; i++) {
				tBias[layer][i] = derivatives[i];
				for (int x = 0; x < weights[layer].length; x++) {
					tWeights[layer][x][i] = derivatives[i] * input[x];
				}
			}
			return this.backprop.func(tWeights, tBias, derivatives, input, --layer);
		} else {
			double[] vector = new double[bias[layer - 1].length];
			for (int x = 0; x < weights[layer].length; x++) {
				double sum = 0;
				for (int i = 0; i < weights[layer][0].length; i++) {
					tBias[layer][i] = derivatives[i];
					tWeights[layer][x][i] = derivatives[i] * sigmoid(z[layer - 1][x]);
					sum += derivatives[i] * weights[layer][x][i];
				}
				vector[x] = sum * sigmoidPrime(z[layer - 1][x]);
			}
			return this.backprop.func(tWeights, tBias, vector, input, --layer);
		}
	};

	public void train(int epochs, int batchSize, List<Pair<double[], double[]>> data, double l_rate) {
		for (int j = 0; j < epochs; j++) {
			List<Pair<double[][][], double[][]>> gradient = new ArrayList<>();
			for (int k = 0; k < data.size(); k++) {
				double[] input = data.get(k).getA();
				double[] y = data.get(k).getB();
				double[] t = predict(input);
				double[] derivatives = new double[bias[bias.length - 1].length];
				for (int i = 0; i < derivatives.length; i++) {
					derivatives[i] = costPrime(t[i], y[i]) * sigmoidPrime(z[z.length - 1][i]);
				}
				gradient.add(backprop.func(init(in, new double[in.length - 1][][]),
						init(in, new double[in.length - 1][]), derivatives, input, bias.length - 1));
			}
			apply(gradient, l_rate);
		}
	}

	private void apply(Pair<double[][][], double[][]> p, double l_rate) {
		double[][][] w = p.getA();
		double[][] b = p.getB();
		for (int layer = 0; layer < w.length; layer++) {
			for (int i = 0; i < w[layer][0].length; i++) {
				bias[layer][i] -= (b[layer][i]) * l_rate;
				for (int x = 0; x < w[layer].length; x++) {
					weights[layer][x][i] -= (w[layer][x][i]) * l_rate;
				}
			}
		}
	}

	private void apply(List<Pair<double[][][], double[][]>> gradient, double l_rate) {
		for (int i = 0; i < gradient.size(); i++) {
			apply(gradient.get(i), l_rate);
		}
	}

	public double[][][] getWeights() {
		return weights;
	}

	public double[][] getBias() {
		return bias;
	}

	public double[][] getZ() {
		return z;
	}

	public double[] predict(double[] x) {
		return predict.func(x, 0);
	}

	private double costPrime(double x, double y) {
		return 2 * (x - y);
	}

	private double cost(double[] x, double[] y) {
		double sum = 0; 
		for (int i = 0 ; i < x.length; i++) {
			sum += Math.pow(x[i] - y[i], 2);	
		}
		return sum;
	}

	private double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}

	private double sigmoidPrime(double x) {
		return Math.pow(Math.E, -x) / Math.pow(1 + Math.pow(Math.E, -x), 2);
	}
}