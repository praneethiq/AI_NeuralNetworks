

public class AITravelGenieMain {
	
	public static void main(String[] args) {
		System.out.println("Welcome to AI Travel Genie\n");
		AITravelGenieNeuralNetwork ain =new AITravelGenieNeuralNetwork();
		ain.learningRate = 0.1;
		ain.momentum = 0.2;
		ain.neurons =5;
		ain.runs =1;
		ain.initializeProgram();
		ain.saveNeuralNetwork("savednetwork_1.nn");
		//Separate agent for the updated neural network
		System.out.println("\nSecond Network with India added as Destination and Hotels added as Feature");
		AITravelGenieNeuralNetwork ain1 =new AITravelGenieNeuralNetwork();
		ain1.learningRate = 0.3;
		ain1.momentum = 0.3;
		ain1.neurons =8;
		ain1.runs =1;
		ain1.addNewDestination();
		ain1.addNewFeature();
		ain1.initializeProgram();
		ain1.saveNeuralNetwork("savednetwork_2.nn");

	}

}
