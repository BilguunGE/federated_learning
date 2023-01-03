import socket
import pickle
import keras
import pygad.kerasga
import pygad

class EdgeNode:
    def __init__(self, training_data, target_data) -> None:
        self.training_data = training_data
        self.target_data = target_data
        self.soc = socket.socket()
        print('Socket is created.')
        self.soc.connect(("localhost", 10000))
        print('Connected to the server.')
        self.init_model()

    def receive_data(self):
        received_data = b''
        while str(received_data)[-2] != '.':
            data = self.soc.recv(1024)
            received_data += data
        received_data = pickle.loads(received_data)
        return received_data

    def init_model(self):
        msg = 'init_model'
        msg = pickle.dumps(msg)
        self.soc.sendall(msg)
        print('Client requested global model')
        received_data = self.receive_data()
        model = received_data.get('model')
        print('Received model from the server')
        self.model = keras.models.clone_model(model)
            
    def init_weights(self):
        msg = 'init_weights'
        msg = pickle.dumps(msg)
        self.soc.sendall(msg)
        print('Client requested global model weights')
        received_data = self.receive_data()
        weights = received_data.get('weights')
        print('Received weights from the server')
        self.model.set_weights(weights)

    def train_model(self):
        self.init_weights()
        keras_ga = pygad.kerasga.KerasGA(model=self.model,
                                 num_solutions=6)
        num_generations = 50
        num_parents_mating = 4 
        initial_population = keras_ga.population_weights
        mutation_percent_genes = 5 
        parent_selection_type = "sss" 
        crossover_type = "single_point" 
        mutation_type = "random" 
        keep_parents = 1 

        init_range_low = -2
        init_range_high = 5

        def fitness_func(solution, sol_idx):
            predictions = pygad.kerasga.predict(model=self.model,
                                                solution=solution,
                                                data=self.training_data)

            mse = keras.losses.MeanSquaredError()
            solution_fitness = 1.0 / (mse(self.target_data, predictions).numpy() + 0.00000001)

            return solution_fitness

        def callback_generation(ga_instance):
            print("Generation = {generation}".format(generation=ga_instance.generations_completed))
            print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))

        ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        mutation_percent_genes=mutation_percent_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        keep_parents=keep_parents,
                        fitness_func=fitness_func,
                        on_generation=callback_generation)


        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        weights = pygad.kerasga.model_weights_as_matrix(model=self.model, weights_vector=solution)

        msg = pickle.dumps(weights)
        self.soc.sendall(msg)
        print('Client send updated weights to the server')






