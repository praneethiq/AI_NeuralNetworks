import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.simple.EncogUtility;

/**
 * This class is for configuring the training table, construct the neural
 * network, train it and test it
 * 
 * @author 180029539
 */
public class AITravelGenieNeuralNetwork {

	List<String[]> choicesTable = new ArrayList<String[]>();
	List<double[]> inputs = new ArrayList<double[]>();
	List<double[]> outputs = new ArrayList<double[]>();

	HashMap<String, double[]> destinationEncodings = new HashMap<String, double[]>();
	String[] headers;
	double[][] trainingInputs, testingInputs;
	String[] destinations;
	String[] uniqueDestinations;
	int counter = 0;
	double[][] trainingOutputs, testingOutputs;
	int totalRows = 0;
	int trainingRows = 0;
	double learningRate = 0.1;
	double momentum = 0.2;
	int runs = 1;
	int neurons = 2;
	double classificationError, avg = 0;
	double trainingRatio = 0.6;
	BasicNetwork network, loaded_net;

	AITravelGenieNeuralNetwork() {
		processFile();
	}

	/**
	 * This method is the point of execution for the class
	 */
	public void initializeProgram() {

		getConfiguration();
		for (int i = 0; i < runs; i++) {
			initiateNeuralNetwork();
			trainNeuralNetwork();
			testNeuralNetwork();

			avg = classificationError + avg;
		}
		// System.out.println(avg / (double) runs);
		// testNeuralNetwork(testingInputs[0]);
	}

	/**
	 * This method takes the file from the project directory and stores the data in
	 * a string array
	 * 
	 * @throws URISyntaxException
	 */
	public void processFile() {

		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(ClassLoader.getSystemClassLoader().getResourceAsStream("trip.csv")))) {
			String line = br.readLine();
			while (line != null) {
				String[] choices = line.split(",");

				choicesTable.add(choices);
				line = br.readLine();
			}
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
		System.out.println("Initial Data Set Processed ");
	}

	/**
	 * This method controls the configuration of the training table
	 */
	public void getConfiguration() {
		processHeaders();
		processInputs();
		processOutputs();
		splitTrainingSet();

	}

	/**
	 * This method takes a destination name and returns how many rows are present in
	 * the dataset
	 * 
	 * @param string Destination to be checked
	 * @return no of occurrences
	 */
	private int calculateDestinationRows(String string) {
		int tempCount = 0;
		for (int i = 0; i < destinations.length; i++) {
			if (destinations[i].equals(string))
				tempCount++;
		}
		return tempCount;
	}

	/**
	 * This takes the headers of the dataset and stores it in a string array
	 */
	private void processHeaders() {
		headers = new String[choicesTable.get(0).length];
		headers = choicesTable.get(0);
	}

	/**
	 * This takes the dataset and generates a list with only inputs
	 */
	private void processInputs() {
		for (int i = 1; i < (choicesTable.size()); i++) {
			inputs.add(processInputStatus(choicesTable.get(i)));
		}
		System.out.println("Inputs Processed. Total No. of Rows = " + inputs.size());
	}

	/**
	 * This reads each entry in the dataset and replaces Yes/No and replaces with
	 * 1.0/0.0
	 * 
	 * @param stringArray The entry to be processed
	 * @return the resulting 1.0/0.0
	 */
	private double[] processInputStatus(String[] stringArray) {
		double[] tempDoubleArray = new double[choicesTable.get(0).length - 1];
		for (int i = 0; i < (choicesTable.get(i).length - 1); i++) {
			double tempDouble = 0.0;
			if (stringArray[i].toLowerCase().equals("yes"))
				tempDouble = 1.0;
			else
				tempDouble = 0.0;
			tempDoubleArray[i] = tempDouble;
		}

		return tempDoubleArray;
	}

	/**
	 * This method handles all the operations required to process the outputs
	 */
	private void processOutputs() {
		processDestinations();
		getUniqueDestinations();
		generateDestinationEncodings();
		convertDestinationsToOutputs();
		System.out.println("Ouputs Processed. Total No. of Rows = " + outputs.size());
	}

	/**
	 * This separates the destinations column into a separate string array from
	 * dataset
	 */
	private void processDestinations() {
		destinations = new String[choicesTable.size() - 1];
		for (int i = 1; i < (choicesTable.size()); i++) {
			destinations[i - 1] = choicesTable.get(i)[choicesTable.get(i).length - 1];
		}
	}

	/**
	 * This method takes the string array with the destinations and gets the array
	 * of unique list of destinations
	 */
	private void getUniqueDestinations() {
		Set<String> tempSet = new HashSet<String>(Arrays.asList(destinations));
		int index = 0;
		uniqueDestinations = new String[tempSet.size()];
		for (String s : tempSet) {
			uniqueDestinations[index++] = s;
		}
		System.out.println("\nNo. of Unique Destinations found = " + uniqueDestinations.length);
	}

	/**
	 * This takes the array of unique destinations and generates the encodings of
	 * outputs
	 */
	private void generateDestinationEncodings() {

		counter = (int) Math.pow(2, (uniqueDestinations.length - 1));

		for (int i = 0; i < uniqueDestinations.length; i++) {
			destinationEncodings.put(uniqueDestinations[i], generateEncoding(uniqueDestinations[i]));
		}

	}

	/**
	 * This method takes in the name of destination and converts it into the
	 * encoding format
	 * 
	 * @param string destination
	 * @return array with encoding
	 */
	private double[] generateEncoding(String string) {

		double[] tempDoubleArray = new double[uniqueDestinations.length];
		if (destinationEncodings.containsKey(string)) {
			tempDoubleArray = destinationEncodings.get(string);
		} else {
			int tempNum = counter;
			for (int i = tempDoubleArray.length - 1; i >= 0; i--) {
				tempDoubleArray[i] = 0.0;
				if (tempNum > 0) {
					if ((tempNum % 2) == 1)
						tempDoubleArray[i] = 1.0;
					tempNum = tempNum / 2;
				}

			}
			counter = counter >> 1;
		}

		return tempDoubleArray;
	}

	/**
	 * This method takes the mappings of destination and their encodings and uses it
	 * to convert the destination column into encoded outputs
	 */
	private void convertDestinationsToOutputs() {

		for (int i = 0; i < destinations.length; i++) {
			outputs.add(destinationEncodings.get(destinations[i]));
		}
		System.out.println("Using One-Hot Encoding.....");
		System.out.println("Destinations Encoded to Generate Ouputs");
	}

	/**
	 * This method splits the input and output lists into training and testing
	 * datasets based on the split ratio
	 */
	private void splitTrainingSet() {
		System.out.println("\nSplitting Data into Training and Testing Sets using " + (int) (trainingRatio * 100) + "/"
				+ (int) ((1 - trainingRatio) * 100) + " Split.");
		List<double[]> trainingInputsList = new ArrayList<double[]>();
		List<double[]> trainingOutputsList = new ArrayList<double[]>();
		List<double[]> testingInputsList = new ArrayList<double[]>();
		List<double[]> testingOutputsList = new ArrayList<double[]>();
		int rows = 0;
		for (int i = 0; i < uniqueDestinations.length; i++) {
			rows = 0;

			totalRows = calculateDestinationRows(uniqueDestinations[i]);
			trainingRows = (int) (totalRows * trainingRatio);
			for (int j = 0; j < destinations.length; j++) {

				if (destinations[j].equals(uniqueDestinations[i])) {
					if (rows <= trainingRows) {
						trainingInputsList.add(inputs.get(j));
						trainingOutputsList.add(outputs.get(j));
					} else {
						testingInputsList.add(inputs.get(j));
						testingOutputsList.add(outputs.get(j));
					}
					rows++;
				}

			}

		}
		trainingInputs = convertListToArray(trainingInputsList);
		trainingOutputs = convertListToArray(trainingOutputsList);
		testingInputs = convertListToArray(testingInputsList);
		testingOutputs = convertListToArray(testingOutputsList);
		System.out.println("Final No. of Rows in Training Set = " + trainingInputs.length);
		System.out.println("Final No. of Rows in Testing Set = " + testingInputs.length);
	}

	/**
	 * This Method converts list to an array
	 * 
	 * @param inputList list
	 * @return array
	 */
	private double[][] convertListToArray(List<double[]> inputList) {
		double[][] tempDoubleArray = new double[inputList.size()][inputList.get(0).length];
		for (int i = 0; i < inputList.size(); i++) {
			tempDoubleArray[i] = inputList.get(i);
		}

		return tempDoubleArray;
	}

	/**
	 * This method initializes a neural network
	 */
	private void initiateNeuralNetwork() {
		System.out.println("\nCreating Neural Network...");
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, false, headers.length - 1));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, neurons));
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, uniqueDestinations.length));
		network.getStructure().finalizeStructure();
		network.reset();
		System.out.println("Neural Network Created With Below Parameters");
		System.out.println(
				"Hidden-Layer Neurons = " + neurons + " Momentum = " + momentum + " Learning Rate = " + learningRate);

	}

	/**
	 * This method trains the neural network based on the training dataset
	 */
	private void trainNeuralNetwork() {
		System.out.println("\nStarted Training Neural Network...");
		MLDataSet trainingSet = new BasicMLDataSet(trainingInputs, trainingOutputs);
		Backpropagation train = new Backpropagation(network, trainingSet, learningRate, momentum);
		int epoch = 1;
		do {
			train.iteration();
			// System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		} while (train.getError() > 0.01 && epoch < 100000);
		train.finishTraining();
		System.out.println("Neural Network Trained in " + epoch + " epochs. Final Error Rate is " + train.getError());
	}

	/**
	 * This method saves the current neural network in the project directory
	 * 
	 * @param string
	 */
	public void saveNeuralNetwork(String string) {
		System.out.println("\nSaving Current Settings...");
		EncogDirectoryPersistence.saveObject(new File(string), network);
		System.out.println("Neural Network Saved with the name '" + string + "'.");
	}

	/**
	 * This method tests the neural network with the test data set and shows the
	 * accuracy of classification
	 */
	private void testNeuralNetwork() {

		MLDataSet testSet = new BasicMLDataSet(testingInputs, testingOutputs);
		classificationError = EncogUtility.calculateClassificationError(network, testSet);
		System.out.println("Accuracy = " + classificationError);

	}

	/**
	 * This method takes a particular input and gives the expected destination
	 * 
	 * @param testInputs
	 */
	@SuppressWarnings("unused")
	private void testNeuralNetwork(double[] testInputs) {
		int outputIndex = 0;
		double currentMaxValue = 0;
		double[] h = testInputs;
		MLData data = new BasicMLData(h);
		MLData output = network.compute(data);
		double[] outputEncoding = new double[output.size()];
		for (int j = 0; j < output.size(); j++) {
			if (currentMaxValue < output.getData(j)) {
				currentMaxValue = output.getData(j);
				outputIndex = j;
			}

		}
		outputEncoding[outputIndex] = 1.0;

		for (Entry<String, double[]> entry : destinationEncodings.entrySet()) {
			if (Arrays.equals(entry.getValue(), outputEncoding)) {
				System.out.println(entry.getKey());
			}
		}

	}

	/**
	 * This method adds India along with random entries into the initial dataset
	 */
	public void addNewDestination() {
		String[][] newDestinationData = { { "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "India" },
				{ "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "India" },
				{ "Yes", "No", "No", "No", "No", "Yes", "Yes", "Yes", "India" },
				{ "Yes", "No", "No", "Yes", "Yes", "Yes", "No", "Yes", "India" },
				{ "No", "Yes", "Yes", "No", "No", "No", "Yes", "Yes", "India" },
				{ "No", "No", "Yes", "No", "No", "Yes", "No", "Yes", "India" },
				{ "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "India" },
				{ "No", "No", "No", "No", "No", "Yes", "Yes", "Yes", "India" },
				{ "Yes", "No", "Yes", "No", "No", "Yes", "Yes", "No", "India" },
				{ "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "India" },
				{ "No", "No", "No", "Yes", "Yes", "No", "No", "No", "India" },
				{ "No", "No", "Yes", "No", "No", "Yes", "No", "No", "India" },
				{ "No", "Yes", "No", "No", "Yes", "No", "No", "No", "India" },
				{ "No", "Yes", "No", "No", "Yes", "No", "No", "Yes", "India" },
				{ "Yes", "No", "No", "No", "No", "Yes", "No", "Yes", "India" },
				{ "No", "Yes", "No", "Yes", "No", "No", "No", "Yes", "India" },
				{ "No", "Yes", "No", "No", "Yes", "No", "No", "No", "India" },
				{ "Yes", "No", "No", "No", "No", "Yes", "Yes", "Yes", "India" } };
		for (int i = 0; i < newDestinationData.length; i++)
			choicesTable.add(newDestinationData[i]);
	}

	/**
	 * This method adds a new feature to the initial dataset
	 */
	public void addNewFeature() {
		List<String[]> tempList = new ArrayList<String[]>();
		tempList.add(concatFeature(choicesTable.get(0), "Hotels"));

		for (int i = 1; i < choicesTable.size(); i++) {
			if (i % 2 == 0)
				tempList.add(concatFeature(choicesTable.get(i), "Yes"));
			else
				tempList.add(concatFeature(choicesTable.get(i), "No"));
		}
		choicesTable = tempList;
	}

	/**
	 * This method concatenates the new feature entry into the existing row in the
	 * dataset
	 * 
	 * @param oldArray
	 * @param newFeature
	 * @return
	 */
	public String[] concatFeature(String[] oldArray, String newFeature) {
		String[] newArray = new String[oldArray.length + 1];
		newArray[0] = newFeature;
		for (int j = 1; j < newArray.length; j++)
			newArray[j] = oldArray[j - 1];
		return newArray;
	}

}
