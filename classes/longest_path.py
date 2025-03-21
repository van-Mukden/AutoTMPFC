import os
import cv2
import numpy as np
import networkx as nx
import csv

class LongestPathOverlay:
    def __init__(self, image_path):
        """
        Initialize with the path to the skeleton image.
        """
        self.image_path = image_path
        self.binary_image = None
        self.longest_path = []
        self.max_length = 0
        self.colored_image = None

    def load_and_preprocess_image(self):
        """
        Load the image in grayscale and convert it to a binary image.
        """
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Image not found or could not be loaded: {self.image_path}")
        # Convert to binary using thresholding.
        _, self.binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    def create_graph(self):
        """
        Create a graph from the white pixels (foreground) in the binary image.
        Each white pixel is a node and edges are added for 8-connected neighbors.
        """
        y_coords, x_coords = np.nonzero(self.binary_image)
        pixels = list(zip(x_coords, y_coords))
        G = nx.Graph()
        for pixel in pixels:
            G.add_node(pixel)
            for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1),
                           (-1, 0), (0, -1), (-1, -1), (-1, 1)]:
                neighbor = (pixel[0] + dx, pixel[1] + dy)
                if neighbor in pixels:
                    G.add_edge(pixel, neighbor)
        return G

    def find_longest_path(self, G):
        """
        Find the longest shortest path (graph diameter) in the graph.
        For each node, the method computes the farthest node and keeps the maximum.
        """
        for node in G.nodes():
            lengths = nx.single_source_shortest_path_length(G, node)
            farthest_node, length = max(lengths.items(), key=lambda x: x[1])
            if length > self.max_length:
                self.max_length = length
                self.longest_path = nx.shortest_path(G, node, farthest_node)

    def overlay_longest_path(self):
        """
        Overlay the longest path onto a colored version of the binary image.
        The longest path is drawn in red (using circles of radius 2 pixels).
        """
        self.colored_image = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
        for x, y in self.longest_path:
            cv2.circle(self.colored_image, (x, y), 2, (0, 0, 255), -1)

    def save_image(self):
        """
        Save the processed image in an "overlaid path" folder.
        For an input image located at:
          .../patient_folder/completed_skeletons/image.png
        the output image is saved at:
          .../patient_folder/overlaid path/image.png
        """
        # Move two levels up: from "completed_skeletons" to the patient folder.
        parent_directory = os.path.dirname(os.path.dirname(self.image_path))
        output_directory = os.path.join(parent_directory, 'overlaid path')
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(output_directory, os.path.basename(self.image_path))
        cv2.imwrite(output_path, self.colored_image)
        print(f"Longest path overlaid image saved at: {output_path}")

    def process(self):
        """
        Run the complete processing pipeline: load image, build graph, find and overlay longest path, and save the result.
        """
        self.load_and_preprocess_image()
        G = self.create_graph()
        self.find_longest_path(G)
        print("Longest path length:", self.max_length)
        self.overlay_longest_path()
        self.save_image()


def store_longest_path_csv(patient_path, data):
    """
    Write a CSV file listing the longest path lengths for each frame.
    The CSV file is saved directly in the patient folder (i.e. the parent directory of the "overlaid path" folder).

    Args:
        patient_path (str): Path to the patient folder.
        data (list of tuples): Each tuple is (filename, longest_path_length).
    """
    csv_file_path = os.path.join(patient_path, 'longest_path_lengths.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "longest_path_length"])
        for filename, length in data:
            writer.writerow([filename, length])
    print(f"CSV file of longest path lengths saved at: {csv_file_path}")
