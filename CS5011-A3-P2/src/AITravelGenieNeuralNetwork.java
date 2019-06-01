import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.*;
import org.encog.persist.EncogDirectoryPersistence;

public class AITravelGenieNeuralNetwork {

	List<String[]> choicesTable = new ArrayList<String[]>();

	HashMap<String, double[]> destinationEncodings = new HashMap<String, double[]>();
	String[] headers;
	String[] destinations;
	String[] uniqueDestinations;
	int counter = 0;
	double[] userInputs;
	BasicNetwork loaded_network;
	Scanner sc1 = new Scanner(System.in);
	int decision = 0;

	AITravelGenieNeuralNetwork() {
	}

	/**
	 * This method is the point of execution for the class
	 * 
	 *
	 */
	public void initializeProgram() {

		System.out.println("Welcome to AI Travel Genie");
		processFile();
		getConfiguration();
		loadNeuralNetwork();
		takeUserInput();

	}

	/**
	 * This method takes the file from the project directory and stores the data in
	 * a string array
	 *
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
	}

	/**
	 * This method controls the configuration of the training table
	 */
	public void getConfiguration() {
		processHeaders();
		processDestinations();

	}

	/**
	 * This takes the headers of the dataset and stores it in a string array
	 */
	private void processHeaders() {
		headers = new String[choicesTable.get(0).length];
		headers = choicesTable.get(0);
	}

	/**
	 * This method handles all the operations required to process the destinations
	 */
	private void processDestinations() {
		getDestinations();
		getUniqueDestinations();
		generateDestinationEncodings();
	}

	/**
	 * This separates the destinations column into a separate string array from
	 * dataset
	 */
	private void getDestinations() {
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
	 * This method loads the neural network from the project directory
	 * 
	 */
	private void loadNeuralNetwork() {
		System.out.println("\nLoading Neural Network...");
		loaded_network = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File("savednetwork_1.nn"));
		System.out.println("Neural Network Loaded from " + System.getProperty("user.dir") + ", with the name '"
				+ "savednetwork_1.nn" + "'.");
	}

	/**
	 * This method takes the user input whether he wants to see a guess after each
	 * answer or all answers
	 */
	private void takeDecision() {
		int num;
		System.out.println("Would you like to 1)Answer all questions 2)Want us to guess after each answer");
		System.out.print("Your Response (Enter 1 for Yes, 2 for No): ");
		do {

			while (!sc1.hasNextInt()) {
				System.out.print("Enter a valid integer choice: ");
				sc1.next();
			}
			num = sc1.nextInt();

			if (!(num == 1 || num == 2))
				System.out.print("Enter a valid integer choice between 1 and 2: ");

		} while (!(num == 1 || num == 2));
		decision = num;
	}

	/**
	 * This method takes a particular input and gives the expected destination
	 * 
	 * @param testInputs
	 */
	private boolean testNeuralNetwork(double[] testInputs) {
		int outputIndex = 0;
		double currentMaxValue = 0;
		double[] h = testInputs;
		MLData data = new BasicMLData(h);
		MLData output = loaded_network.compute(data);
		double[] outputEncoding = new double[output.size()];
		for (int j = 0; j < output.size(); j++) {
			if (currentMaxValue < output.getData(j)) {
				currentMaxValue = output.getData(j);
				outputIndex = j;
			}

		}
		outputEncoding[outputIndex] = 1.0;

		return verifyResult(outputEncoding);

	}

	/**
	 * This method asks the user whether the guess is correct or not
	 * 
	 * @param outputEncoding the output from the neural network
	 * @return
	 */
	private boolean verifyResult(double[] outputEncoding) {

		String dreamDestination = "";
		for (Entry<String, double[]> entry : destinationEncodings.entrySet()) {
			if (Arrays.equals(entry.getValue(), outputEncoding)) {
				dreamDestination = entry.getKey();
			}
		}
		int result = 0;
		System.out.println("\nIs your Dream Destination " + dreamDestination + "?");
		System.out.print("Your Response (Enter 1 for Yes, 2 for No): ");
		do {

			while (!sc1.hasNextInt()) {
				System.out.print("Enter a valid integer choice: ");
				sc1.next();
			}
			result = sc1.nextInt();

			if (!(result == 1 || result == 2))
				System.out.print("Enter a valid integer choice between 1 and 2: ");

		} while (!(result == 1 || result == 2));
		if (result == 1)
			return true;
		else
			return false;
	}

	/**
	 * This method takes the user input for each question
	 */
	private void takeUserInput() {
		int choice = 0;
		userInputs = new double[(headers.length - 1)];
		System.out.println("\nAnswer " + (headers.length - 1) + " Questions about your Dream Destination.");
		System.out.println("We Will Try to Guess the Destination.");
		System.out.println("Let's get Started...");

		takeDecision();

		for (int i = 0; i < (headers.length - 1); i++) {
			System.out.println("\nQuestion " + (i + 1) + " : ");
			System.out.println("Is your Dream Destination known for " + headers[i] + "?");
			System.out.print("Your Response (Enter 1 for Yes, 2 for No): ");
			do {

				while (!sc1.hasNextInt()) {
					System.out.print("Enter a valid integer choice: ");
					sc1.next();
				}
				choice = sc1.nextInt();

				if (!(choice == 1 || choice == 2))
					System.out.print("Enter a valid integer choice between 1 and 2: ");

			} while (!(choice == 1 || choice == 2));
			userInputs[i] = (double) (2 - choice);
			if (decision == 2) {
				if (testNeuralNetwork(userInputs))
					break;
			}
		}
		if (decision == 1)
			testNeuralNetwork(userInputs);
	}

}
