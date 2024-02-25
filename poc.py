import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return int(np.round(x0)), int(np.round(y0))

image_path = './testimg/test3.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 200, 255, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
main_vanishing_point = None
intersections = []
coords = []

if lines is not None:
    for i in range(len(lines)):
        rho, theta = lines[i][0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (100, 0, 0), 2)

        for j in range(i + 1, len(lines)):
            line1 = lines[i][0]
            line2 = lines[j][0]
            try:
                intersection = line_intersection(line1, line2)
                if 0 <= intersection[0] < image.shape[1] and 0 <= intersection[1] < image.shape[0]:
                    intersections.append(intersection)
                    cv2.circle(image, intersection, 5, (0, 150, 0), -1)
            except np.linalg.LinAlgError:
                continue

if intersections:
    intersections = np.array(intersections)
    clustering = DBSCAN(eps=50, min_samples=10).fit(intersections)
    labels = clustering.labels_
    
    if len(set(labels)) > 1:
        largest_cluster = max(set(labels), key=list(labels).count)
        main_vanishing_point = np.mean(intersections[labels == largest_cluster], axis=0)
        main_vanishing_point = tuple(main_vanishing_point.astype(int))
        cv2.circle(image, main_vanishing_point, 25, (0, 255, 0), -1)
    else:
        print("No clear vanishing point found. Try adjusting DBSCAN parameters.")

def find_intersection(p1, p2, p3, p4):
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]
    
    determinant = a1 * b2 - a2 * b1
    
    if determinant == 0:
        return None
    else:
        x = (c1 * b2 - c2 * b1) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return (int(x), int(y))

def onclick(event):
    global image, coords
    if event.button == 1:
        coords.append((int(event.xdata), int(event.ydata)))
        cv2.circle(image, coords[-1], 5, (0, 0, 200), -1)

        if len(coords) == 2:
            cv2.line(image, coords[0], coords[1], (0, 255, 255), 2)
        elif len(coords) == 4:
            cv2.line(image, coords[2], coords[3], (0, 255, 255), 2)
            intersection = find_intersection(coords[0], coords[1], coords[2], coords[3])
            if intersection:
                cv2.circle(image, intersection, 15, (0, 0, 255), -1)
            coords = []

        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.draw()

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()