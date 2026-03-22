from __future__ import print_function

import cv2
import numpy as np

import aslam_cv as acv
import sm

try:
    import onnxruntime as ort
    _ORT_IMPORT_ERROR = None
except Exception as e:
    ort = None
    _ORT_IMPORT_ERROR = e


class FisheyeAprilgridOnnxDetector(object):
    _CORNER_REMAP = [1, 0, 3, 2]

    def __init__(self,
                 cameraGeometry,
                 grid,
                 tagRows,
                 tagCols,
                 modelPath,
                 heatmapThreshold=0.7,
                 minCornersPerTag=3,
                 minBorderDistance=4.0,
                 minTagsForValidObs=7,
                 showExtractionVideo=False,
                 imageStepping=False):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is required for detectorBackend=fisheye_onnx. "
                "Please install it in the Noetic environment. Import error: {0}".format(_ORT_IMPORT_ERROR)
            )

        self._cameraGeometry = cameraGeometry
        self._grid = grid
        self._tagRows = int(tagRows)
        self._tagCols = int(tagCols)
        self._numTags = int(self._tagRows * self._tagCols)
        self._modelPath = str(modelPath)
        self._heatmapThreshold = float(heatmapThreshold)
        self._minCornersPerTag = int(minCornersPerTag)
        self._minBorderDistance = float(minBorderDistance)
        self._minTagsForValidObs = int(minTagsForValidObs)
        self._showExtractionVideo = bool(showExtractionVideo)
        self._imageStepping = bool(imageStepping)
        self.supportsMultithreading = False

        try:
            self._session = ort.InferenceSession(
                self._modelPath,
                providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load fisheye ONNX detector from '{0}': {1}".format(self._modelPath, e)
            )

        self._inputName = self._session.get_inputs()[0].name
        self._outputNames = [o.name for o in self._session.get_outputs()]

        if self._showExtractionVideo:
            cv2.namedWindow("Fisheye Aprilgrid corners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Fisheye Aprilgrid corners", 640, 480)

    def target(self):
        return self._grid

    def findTargetNoTransformation(self, stamp, image):
        imageGray = self._normalize_image(image)
        observation = acv.GridCalibrationTargetObservation(self._grid)
        observation.setImage(imageGray)
        observation.setTime(stamp)

        outputs = self._run_inference(imageGray)
        xy_px = self._reshape_xy_px(outputs["xy_px"])
        heatmapConf = self._reshape_heatmap_conf(outputs["heatmap_logits"])

        imageHeight, imageWidth = imageGray.shape[:2]
        gridCols = int(self._grid.cols())
        activeTagCount = 0
        acceptedPoints = []
        rejectedPoints = []

        for tagId in range(self._numTags):
            tagCorners = xy_px[tagId][self._CORNER_REMAP]
            tagHeatmapConf = self._sample_heatmap_conf(
                heatmaps=heatmapConf[tagId][self._CORNER_REMAP],
                corners=tagCorners,
                imageWidth=imageWidth,
                imageHeight=imageHeight
            )
            validCorners = []

            baseId = (tagId // self._tagCols) * gridCols * 2 + (tagId % self._tagCols) * 2
            pointIndices = [baseId, baseId + 1, baseId + gridCols + 1, baseId + gridCols]

            for localIdx, pointIdx in enumerate(pointIndices):
                corner = tagCorners[localIdx]
                cornerValid = self._is_corner_valid(
                    corner=corner,
                    heatmapConf=float(tagHeatmapConf[localIdx]),
                    imageWidth=imageWidth,
                    imageHeight=imageHeight
                )
                validCorners.append(cornerValid)

                pointTuple = (float(corner[0]), float(corner[1]))
                if cornerValid:
                    observation.updateImagePoint(
                        pointIdx,
                        np.asarray([pointTuple[0], pointTuple[1]], dtype=np.float64)
                    )
                    acceptedPoints.append(pointTuple)
                else:
                    if np.isfinite(corner).all():
                        rejectedPoints.append(pointTuple)

            if sum(validCorners) >= self._minCornersPerTag:
                activeTagCount += 1

        success = activeTagCount >= self._minTagsForValidObs

        if self._showExtractionVideo:
            failureReason = None
            if not success:
                failureReason = "valid tags {0}/{1}, required {2}".format(
                    activeTagCount,
                    self._numTags,
                    self._minTagsForValidObs
                )
            self._show_overlay(
                imageGray=imageGray,
                acceptedPoints=acceptedPoints,
                rejectedPoints=rejectedPoints,
                success=success,
                failureReason=failureReason
            )

        return success, observation

    def findTarget(self, stamp, image):
        success, observation = self.findTargetNoTransformation(stamp, image)
        if success:
            poseSuccess, transform = self._cameraGeometry.estimateTransformation(observation)
            if poseSuccess:
                observation.set_T_t_c(transform)
            else:
                sm.logWarn("FisheyeAprilgridOnnxDetector: estimateTransformation() failed")
                success = False

        return success, observation

    def _normalize_image(self, image):
        imageArray = np.asarray(image)
        if imageArray.ndim == 3:
            imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)

        if imageArray.dtype != np.uint8:
            imageArray = np.clip(imageArray, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(imageArray)

    def _prepare_input(self, imageGray):
        rgb = np.repeat(imageGray[:, :, None], 3, axis=2).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(x.astype(np.float32))

    def _run_inference(self, imageGray):
        x = self._prepare_input(imageGray)

        try:
            outVals = self._session.run(None, {self._inputName: x})
        except Exception as e:
            raise RuntimeError("Failed to run fisheye ONNX detector inference: {0}".format(e))

        outputs = dict(zip(self._outputNames, outVals))
        for requiredName in ["xy_px", "heatmap_logits"]:
            if requiredName not in outputs:
                raise RuntimeError(
                    "Fisheye ONNX detector output is missing '{0}'. Available outputs: {1}".format(
                        requiredName, sorted(outputs.keys())
                    )
                )
        return outputs

    def _reshape_xy_px(self, xy_px):
        array = np.asarray(xy_px, dtype=np.float32)
        if array.ndim == 4:
            array = array[0]
        elif array.ndim == 3:
            array = array[0].reshape(self._numTags, 4, 2)
        else:
            raise RuntimeError("Unexpected xy_px shape: {0}".format(array.shape))

        if array.shape != (self._numTags, 4, 2):
            raise RuntimeError(
                "Unexpected xy_px shape after reshape: {0}, expected ({1}, 4, 2)".format(
                    array.shape, self._numTags
                )
            )
        return array

    def _reshape_heatmap_conf(self, heatmap_logits):
        array = np.asarray(heatmap_logits, dtype=np.float32)
        array = self._sigmoid(array)

        if array.ndim == 5:
            array = array[0]
        elif array.ndim == 4:
            array = array[0].reshape(self._numTags, 4, array.shape[-2], array.shape[-1])
        else:
            raise RuntimeError("Unexpected heatmap_logits shape: {0}".format(array.shape))

        if array.shape[0] != self._numTags or array.shape[1] != 4:
            raise RuntimeError(
                "Unexpected heatmap_logits shape after reshape: {0}, expected ({1}, 4, H, W)".format(
                    array.shape, self._numTags
                )
            )
        return array

    def _sigmoid(self, values):
        values = np.asarray(values, dtype=np.float32)
        out = np.empty_like(values, dtype=np.float32)
        pos = values >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
        expv = np.exp(values[~pos])
        out[~pos] = expv / (1.0 + expv)
        return out

    def _sample_heatmap_conf(self, heatmaps, corners, imageWidth, imageHeight):
        numCorners = heatmaps.shape[0]
        heatmapHeight = int(heatmaps.shape[1])
        heatmapWidth = int(heatmaps.shape[2])
        conf = np.zeros((numCorners,), dtype=np.float32)

        denomW = float(max(imageWidth - 1, 1))
        denomH = float(max(imageHeight - 1, 1))
        denomHmW = float(max(heatmapWidth - 1, 1))
        denomHmH = float(max(heatmapHeight - 1, 1))

        for idx in range(numCorners):
            corner = corners[idx]
            if not np.isfinite(corner).all():
                conf[idx] = 0.0
                continue

            ix = int(np.round(float(corner[0]) / denomW * denomHmW))
            iy = int(np.round(float(corner[1]) / denomH * denomHmH))
            ix = int(np.clip(ix, 0, heatmapWidth - 1))
            iy = int(np.clip(iy, 0, heatmapHeight - 1))
            conf[idx] = float(heatmaps[idx, iy, ix])

        return conf

    def _is_corner_valid(self, corner, heatmapConf, imageWidth, imageHeight):
        if not np.isfinite(corner).all():
            return False
        if heatmapConf < self._heatmapThreshold:
            return False

        x = float(corner[0])
        y = float(corner[1])
        border = self._minBorderDistance
        if x < border or x > (float(imageWidth) - border):
            return False
        if y < border or y > (float(imageHeight) - border):
            return False
        return True

    def _show_overlay(self, imageGray, acceptedPoints, rejectedPoints, success, failureReason):
        overlay = cv2.cvtColor(imageGray, cv2.COLOR_GRAY2BGR)

        for point in rejectedPoints:
            if not np.isfinite(point[0]) or not np.isfinite(point[1]):
                continue
            x = int(round(point[0]))
            y = int(round(point[1]))
            cv2.circle(overlay, (x, y), 3, (0, 0, 255), 1)

        for point in acceptedPoints:
            if not np.isfinite(point[0]) or not np.isfinite(point[1]):
                continue
            x = int(round(point[0]))
            y = int(round(point[1]))
            cv2.circle(overlay, (x, y), 3, (255, 0, 0), 1)

        if success:
            cv2.putText(
                overlay,
                "Detection success",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                8,
                False
            )
        else:
            message = "Detection failed"
            if failureReason:
                message = "{0}: {1}".format(message, failureReason)
            cv2.putText(
                overlay,
                message,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                8,
                False
            )

        cv2.imshow("Fisheye Aprilgrid corners", overlay)
        if self._imageStepping:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
