from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

class RedPixelFlowAnalyzer:
    def __init__(self, frame_folder, flow_arrows_dir, flow_data_csv, clustered_frames_dir, cluster_points_csv, eps, min_samples, step):
        self.frame_folder = frame_folder
        self.flow_arrows_dir = flow_arrows_dir
        self.flow_data_csv = flow_data_csv
        self.clustered_frames_dir = clustered_frames_dir
        self.cluster_points_csv = cluster_points_csv  # Path for saving clustered points
        self.eps = eps
        self.min_samples = min_samples
        self._step = step  # Private variable for controlling flow vector density

        # Create output directories if they don't exist
        os.makedirs(self.flow_arrows_dir, exist_ok=True)
        os.makedirs(self.clustered_frames_dir, exist_ok=True)

        # Load frames
        self.frames = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png') or f.endswith('.jpg')])

        # Define red color thresholds in HSV
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        # Optical flow parameters
        self.flow_params = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Data storage for flow results and clusters
        self.flow_data = []
        self.clustered_points = []

    def calculate_flow(self):
        for i in range(len(self.frames) - 1):
            frame1_path = os.path.join(self.frame_folder, self.frames[i])
            frame2_path = os.path.join(self.frame_folder, self.frames[i + 1])

            # Read consecutive frames
            frame1 = cv2.imread(frame1_path)
            frame2 = cv2.imread(frame2_path)

            # Process frames for red pixel flow using the instance variable step
            x, y, fx, fy = self._process_frame_pair(frame1, frame2)

            # Save flow data for this frame pair
            self.flow_data.extend([(i, x[j], y[j], fx[j], fy[j]) for j in range(len(x))])

            # Save flow arrows visualization
            self._save_flow_arrows_visualization(frame1, x, y, fx, fy, i)

            # Cluster and save clustered points and visualization
            if len(x) > 0:  # Only cluster if there are points to cluster
                self._cluster_and_visualize(frame1, x, y, fx, fy, i)
            else:
                print(f"No points to cluster for frame {i}, skipping.")

        # Save flow data to CSV
        self.save_flow_dataframe()
        # Save all clustered points to CSV
        self.save_clustered_points()

    def _process_frame_pair(self, frame1, frame2):
        # Convert frames to HSV
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

        # Create masks for red areas in both frames
        mask1 = cv2.inRange(hsv1, self.lower_red1, self.upper_red1) | cv2.inRange(hsv1, self.lower_red2, self.upper_red2)
        mask2 = cv2.inRange(hsv2, self.lower_red1, self.upper_red1) | cv2.inRange(hsv2, self.lower_red2, self.upper_red2)

        # Apply masks to isolate red areas
        red_frame1 = cv2.bitwise_and(frame1, frame1, mask=mask1)
        red_frame2 = cv2.bitwise_and(frame2, frame2, mask=mask2)

        # Convert red isolated areas to grayscale for optical flow calculation
        gray1 = cv2.cvtColor(red_frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(red_frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow on red pixels only, with adjustable step size
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, **self.flow_params)

        # Set up grid for vector plotting with the instance variable step
        h, w = gray1.shape
        step = self._step
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # Filter out vectors in non-red regions
        red_locations = mask1[y, x] > 0
        x, y, fx, fy = x[red_locations], y[red_locations], fx[red_locations], fy[red_locations]

        return x, y, fx, fy

    def _save_flow_arrows_visualization(self, frame, x, y, fx, fy, frame_index):
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.quiver(x, y, fx, fy, color='cyan', scale=50)
        ax.set_title(f"Frame {frame_index} - Optical Flow Arrows")
        ax.axis('off')

        # Save the plot as an image file
        output_path = os.path.join(self.flow_arrows_dir, f'flow_arrows_{frame_index}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _cluster_and_visualize(self, frame, x, y, fx, fy, frame_index):
        # Cluster based on position only
        data = np.vstack((x, y)).T

        # Perform DBSCAN clustering on position only
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)
        labels = clustering.labels_

        # Collect clustered points
        for label in set(labels):
            if label == -1:
                continue  # Skip noise
            points = data[labels == label]
            for (px, py) in points:
                self.clustered_points.append((frame_index, label, int(px), int(py)))

        # Prepare plot
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Use different colors for each cluster
        unique_labels = set(labels)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                continue  # Skip noise
            cluster_x, cluster_y = data[labels == label][:, 0], data[labels == label][:, 1]
            ax.scatter(cluster_x, cluster_y, color=color, s=20, label=f"Cluster {label}")

        ax.set_title(f"Frame {frame_index} - All Clusters")
        ax.axis('off')
        ax.legend(loc="best", fontsize='small', markerscale=1.5)

        # Save the combined cluster plot for the frame
        output_path = os.path.join(self.clustered_frames_dir, f'clusters_{frame_index}.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def save_flow_dataframe(self):
        df = pd.DataFrame(self.flow_data, columns=['Frame', 'x', 'y', 'fx', 'fy'])
        df.to_csv(self.flow_data_csv, index=False)
        print(f"Flow data saved to {self.flow_data_csv}")

    def save_clustered_points(self):
        df_clusters = pd.DataFrame(self.clustered_points, columns=['Frame', 'Cluster', 'x', 'y'])
        df_clusters.to_csv(self.cluster_points_csv, index=False)
        print(f"Clustered points saved to {self.cluster_points_csv}")
