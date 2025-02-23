import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class PSOFeatureSelector:
    def __init__(self, X, y, num_particles=10, max_iter=20, alpha=0.99,
                 w=0.7, c1=2.0, c2=2.0):  # Added w, c1, c2 parameters
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.num_features = X.shape[1]
        self.global_best_position = np.random.randint(2, size=self.num_features)
        self.global_best_score = 0
        self.alpha = alpha

        # PSO parameters
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient

        self.history = {
            'global_best_scores': [],
            'mean_particle_scores': [],
            'selected_feature_counts': [],
            'pure_accuracy_scores': []
        }

    def fitness(self, features):
        if np.sum(features) == 0:
            return 0, 0

        selected_features = self.X.iloc[:, features.astype(bool)]
        X_train, X_test, y_train, y_test = train_test_split(selected_features, self.y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Feature count penalty
        feature_ratio = np.sum(features) / self.num_features
        penalty = (1 - feature_ratio)

        score = self.alpha * accuracy + (1 - self.alpha) * penalty

        return score, accuracy

    def optimize(self):
        print("\n=== Starting PSO Feature Selection ===")
        print(f"Dataset dimensions: {self.X.shape}")
        print(f"Parameters: particles={self.num_particles}, iterations={self.max_iter}")
        print(f"PSO parameters: w={self.w}, c1={self.c1}, c2={self.c2}, alpha={self.alpha}")

        # Initialize particles and velocities
        particles = np.random.randint(2, size=(self.num_particles, self.num_features))
        velocities = np.random.rand(self.num_particles, self.num_features) * 2 - 1  # Velocities between -1 and 1

        # Initialize personal bests
        personal_best_positions = np.copy(particles)
        personal_best_scores = []
        personal_best_accuracies = []

        # Evaluate initial positions
        for p in particles:
            score, accuracy = self.fitness(p)
            personal_best_scores.append(score)
            personal_best_accuracies.append(accuracy)

        personal_best_scores = np.array(personal_best_scores)
        personal_best_accuracies = np.array(personal_best_accuracies)

        # Initialize global best
        best_idx = np.argmax(personal_best_scores)
        self.global_best_position = personal_best_positions[best_idx]
        self.global_best_score = personal_best_scores[best_idx]
        self.global_best_accuracy = personal_best_accuracies[best_idx]

        print(f"\nInitial state:")
        print(f"Best score: {self.global_best_score:.4f}")
        print(f"Pure accuracy: {self.global_best_accuracy:.4f}")
        print(f"Selected features: {np.sum(self.global_best_position)}")

        for iteration in range(self.max_iter):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iter} ---")

            iteration_scores = []
            iteration_accuracies = []

            for i in range(self.num_particles):
                # Update velocity using standard PSO formula
                inertia = self.w * velocities[i]
                cognitive = self.c1 * np.random.rand(self.num_features) * (personal_best_positions[i] - particles[i])
                social = self.c2 * np.random.rand(self.num_features) * (self.global_best_position - particles[i])

                velocities[i] = inertia + cognitive + social

                # Update position using sigmoid function
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = np.where(np.random.rand(self.num_features) < sigmoid, 1, 0)

                # Evaluate new position
                score, accuracy = self.fitness(particles[i])
                iteration_scores.append(score)
                iteration_accuracies.append(accuracy)

                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_accuracies[i] = accuracy
                    personal_best_positions[i] = particles[i]

                # Update global best
                if score > self.global_best_score:
                    old_features = np.sum(self.global_best_position)
                    new_features = np.sum(particles[i])

                    self.global_best_score = score
                    self.global_best_accuracy = accuracy
                    self.global_best_position = particles[i]

                    print(f"\nNew best solution found:")
                    print(f"Score: {score:.4f}")
                    print(f"Pure accuracy: {accuracy:.4f}")
                    print(f"Features: {new_features} ({new_features / self.num_features * 100:.1f}%)")
                    print(f"Change in features: {old_features} -> {new_features}")

            # Record iteration statistics
            mean_score = np.mean(iteration_scores)
            mean_accuracy = np.mean(iteration_accuracies)
            selected_features_count = np.sum(self.global_best_position)

            self.history['global_best_scores'].append(self.global_best_score)
            self.history['mean_particle_scores'].append(mean_score)
            self.history['selected_feature_counts'].append(selected_features_count)
            self.history['pure_accuracy_scores'].append(self.global_best_accuracy)

            print(f"\nIteration Summary:")
            print(f"Best Score: {self.global_best_score:.4f}")
            print(f"Best Accuracy: {self.global_best_accuracy:.4f}")
            print(f"Selected Features: {selected_features_count}")
            print(f"Mean Score: {mean_score:.4f}")

        print("\n=== PSO Feature Selection Completed ===")
        print(f"Final Best Score: {self.global_best_score:.4f}")
        print(f"Final Accuracy: {self.global_best_accuracy:.4f}")
        print(f"Final Selected Features: {selected_features_count}")

        return self.global_best_position, self.global_best_score, self.global_best_accuracy, self.history


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("../data/BinaryClassify/train_nsl_kdd_binary_encoded.csv")
    X = df.drop(columns=["binaryoutcome"])
    y = df["binaryoutcome"]

    # Initialize with standard PSO parameters
    pso_selector = PSOFeatureSelector(
        X, y,
        num_particles=30,
        max_iter=20,
        alpha=0.99,
        w=0.7,  # Inertia weight
        c1=2.0,  # Cognitive coefficient
        c2=2.0  # Social coefficient
    )

    best_features, best_score, best_accuracy, history = pso_selector.optimize()
    selected_feature_names = X.columns[best_features.astype(bool)]

    print("\nFinal Results:")
    print(f"Selected {len(selected_feature_names)} out of {X.shape[1]} features")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Combined Score: {best_score:.4f}")
    print("\nSelected features:", list(selected_feature_names))