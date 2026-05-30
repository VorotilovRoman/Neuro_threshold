from import_libs_external import *

def apply_gradient_method(gray, method, params):
    if method == "Sobel":
        ksize = params.get("ksize", 3)
        scale = params.get("scale", 1)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        return magnitude

    elif method == "Scharr":
        scale = params.get("scale", 1)
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0, scale=scale)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1, scale=scale)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        return magnitude

    elif method == "Laplacian":
        ksize = params.get("ksize", 3)
        scale = params.get("scale", 1)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        return laplacian

    elif method == "Canny":
        t1 = params.get("threshold1", 50)
        t2 = params.get("threshold2", 150)
        aperture = params.get("aperture", 3)
        edges = cv2.Canny(gray, t1, t2, apertureSize=aperture)
        return edges

    elif method == "Prewitt":
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        return magnitude

    elif method == "Roberts":
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        grad_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))
        return magnitude

    elif method == "Kirsch":
        kernels = [
            np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[ 5,  5, -3], [ 5,  0, -3], [-3, -3, -3]], dtype=np.float32),
            np.array([[ 5, -3, -3], [ 5,  0, -3], [ 5, -3, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [ 5,  0, -3], [ 5,  5, -3]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3,  0, -3], [ 5,  5,  5]], dtype=np.float32),
            np.array([[-3, -3, -3], [-3,  0,  5], [-3,  5,  5]], dtype=np.float32),
            np.array([[-3, -3,  5], [-3,  0,  5], [-3, -3,  5]], dtype=np.float32),
            np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float32)
        ]
        max_response = np.zeros_like(gray, dtype=np.float64)
        for kernel in kernels:
            response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            max_response = np.maximum(max_response, np.abs(response))   # ИСПРАВЛЕНО
        magnitude = np.uint8(np.clip(max_response, 0, 255))
        return magnitude

    else:
        raise ValueError(f"Unknown gradient method: {method}")