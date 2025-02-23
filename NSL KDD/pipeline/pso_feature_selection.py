import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


class PSOFeatureSelector:
    def __init__(self, X, y, num_particles=10, max_iter=20, alpha=0.99,
                 w=0.7, c1=2.0, c2=2.0, v_max=6.0, v_min=-6.0):
        self.X = X
        self.y = y
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.num_features = X.shape[1]
        self.global_best_position = np.random.randint(2, size=self.num_features)
        self.global_best_score = 0
        self.alpha = alpha

        # PSO parameters
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.v_min = v_min

        self.history = {
            'global_best_scores': [],
            'mean_particle_scores': [],
            'selected_feature_counts': [],
            'pure_accuracy_scores': [],
            'iteration_times': []  # Added to track performance over time
        }

    def clip_velocity(self, velocity):
        return np.clip(velocity, self.v_min, self.v_max)

    def initialize_particles(self):
        """Initialize particles ensuring no all-zero vectors"""
        particles = np.random.randint(2, size=(self.num_particles, self.num_features))

        # Check for and fix any all-zero particles
        for i in range(self.num_particles):
            if np.sum(particles[i]) == 0:
                # Randomly set 1-3 features to 1
                num_features_to_set = np.random.randint(1, 4)
                random_features = np.random.choice(self.num_features,
                                                   size=num_features_to_set,
                                                   replace=False)
                particles[i][random_features] = 1

        return particles

    def fitness(self, features):
        if np.sum(features) == 0:
            return 0, 0

        selected_features = self.X.iloc[:, features.astype(bool)]
        X_train, X_test, y_train, y_test = train_test_split(selected_features, self.y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        feature_ratio = np.sum(features) / self.num_features
        penalty = (1 - feature_ratio)
        score = self.alpha * accuracy + (1 - self.alpha) * penalty

        return score, accuracy

    def plot_optimization_history(self, save_path='pso_results'):
        """
        Plot optimization history metrics and save to files
        Args:
            save_path: Directory to save the plots (will be created if doesn't exist)
        """
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Set style for better-looking plots
        plt.style.use('seaborn')

        # Plot 1: Scores over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['global_best_scores'], label='Best Score', color='blue', linewidth=2)
        plt.plot(self.history['mean_particle_scores'], label='Mean Score', color='gray', alpha=0.6)
        plt.title('Score Evolution Over Iterations', pad=20)
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/score_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Pure accuracy over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['pure_accuracy_scores'], label='Best Accuracy', color='green', linewidth=2)
        plt.title('Accuracy Evolution Over Iterations', pad=20)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/accuracy_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Selected feature count over iterations
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['selected_feature_counts'], label='Selected Features',
                 color='red', linewidth=2)
        plt.title('Feature Selection Evolution Over Iterations', pad=20)
        plt.xlabel('Iteration')
        plt.ylabel('Number of Selected Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 4: Score vs Feature Count scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(self.history['selected_feature_counts'],
                    self.history['global_best_scores'],
                    alpha=0.6, c='purple')
        plt.title('Score vs Number of Selected Features', pad=20)
        plt.xlabel('Number of Selected Features')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/score_vs_features.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create a summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Scores
        ax1.plot(self.history['global_best_scores'], label='Best Score', color='blue', linewidth=2)
        ax1.plot(self.history['mean_particle_scores'], label='Mean Score', color='gray', alpha=0.6)
        ax1.set_title('Score Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Accuracy
        ax2.plot(self.history['pure_accuracy_scores'], label='Best Accuracy',
                 color='green', linewidth=2)
        ax2.set_title('Accuracy Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Feature Count
        ax3.plot(self.history['selected_feature_counts'], label='Selected Features',
                 color='red', linewidth=2)
        ax3.set_title('Feature Selection Evolution')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Number of Features')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Scatter
        ax4.scatter(self.history['selected_feature_counts'],
                    self.history['global_best_scores'],
                    alpha=0.6, c='purple')
        ax4.set_title('Score vs Feature Count')
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('Score')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nPlots have been saved to directory: {save_path}")
        print("Files saved:")
        print("- score_evolution.png")
        print("- accuracy_evolution.png")
        print("- feature_evolution.png")
        print("- score_vs_features.png")
        print("- summary_plot.png")

    def optimize(self):
        print("\n=== Starting PSO Feature Selection ===")
        print(f"Dataset dimensions: {self.X.shape}")
        print(f"Parameters: particles={self.num_particles}, iterations={self.max_iter}")
        print(f"PSO parameters: w={self.w}, c1={self.c1}, c2={self.c2}, alpha={self.alpha}")
        print(f"Velocity bounds: [{self.v_min}, {self.v_max}]")

        # Initialize particles with no all-zero vectors
        particles = self.initialize_particles()
        velocities = np.random.uniform(self.v_min, self.v_max,
                                       size=(self.num_particles, self.num_features))

        personal_best_positions = np.copy(particles)
        personal_best_scores = []
        personal_best_accuracies = []

        for p in particles:
            score, accuracy = self.fitness(p)
            personal_best_scores.append(score)
            personal_best_accuracies.append(accuracy)

        personal_best_scores = np.array(personal_best_scores)
        personal_best_accuracies = np.array(personal_best_accuracies)

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
                inertia = self.w * velocities[i]
                cognitive = self.c1 * np.random.rand(self.num_features) * (personal_best_positions[i] - particles[i])
                social = self.c2 * np.random.rand(self.num_features) * (self.global_best_position - particles[i])

                velocities[i] = self.clip_velocity(inertia + cognitive + social)

                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                particles[i] = np.where(np.random.rand(self.num_features) < sigmoid, 1, 0)

                # Ensure no all-zero particles
                if np.sum(particles[i]) == 0:
                    random_feature = np.random.randint(0, self.num_features)
                    particles[i][random_feature] = 1

                score, accuracy = self.fitness(particles[i])
                iteration_scores.append(score)
                iteration_accuracies.append(accuracy)

                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_accuracies[i] = accuracy
                    personal_best_positions[i] = particles[i]

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

        # Plot the optimization history
        self.plot_optimization_history()

        return self.global_best_position, self.global_best_score, self.global_best_accuracy, self.history


# Example usage
if __name__ == "__main__":
    df = pd.read_csv("../data/BinaryClassify/train_nsl_kdd_binary_encoded.csv")
    X = df.drop(columns=["binaryoutcome"])
    y = df["binaryoutcome"]

    pso_selector = PSOFeatureSelector(
        X, y,
        num_particles=30,
        max_iter=500,
        alpha=0.5,
        w=0.9,
        c1=2.0,
        c2=2.0,
        v_max=4.0,
        v_min=-4.0
    )

    best_features, best_score, best_accuracy, history = pso_selector.optimize()
    selected_feature_names = X.columns[best_features.astype(bool)]

    print("\nFinal Results:")
    print(f"Selected {len(selected_feature_names)} out of {X.shape[1]} features")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"Combined Score: {best_score:.4f}")
    print("\nSelected features:", list(selected_feature_names))