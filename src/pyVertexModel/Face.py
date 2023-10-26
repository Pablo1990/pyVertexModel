import numpy as np


class Face():
    def __int__(self):
        pass

    def ComputeTriAspectRatio(self, sideLengths):
        s = np.sum(sideLengths) / 2
        aspectRatio = (sideLengths[0] * sideLengths[1] * sideLengths[2]) / (
                    8 * (s - sideLengths[0]) * (s - sideLengths[1]) * (s - sideLengths[2]))
        return aspectRatio

    def ComputeTriLengthMeasurements(self, Tris, Ys, currentTri, FaceCentre):
        EdgeLength = np.linalg.norm(Ys[Tris[currentTri].Edge[0], :] - Ys[Tris[currentTri].Edge[1], :])
        LengthsToCentre = [np.linalg.norm(Ys[Tris[currentTri].Edge[0], :] - FaceCentre),
                           np.linalg.norm(Ys[Tris[currentTri].Edge[1], :] - FaceCentre)]
        AspectRatio = self.ComputeTriAspectRatio([EdgeLength] + LengthsToCentre)
        return EdgeLength, LengthsToCentre, AspectRatio

    def ComputeFaceEdgeLengths(self, Face, Ys):
        EdgeLength = []
        LengthsToCentre = []
        AspectRatio = []
        for currentTri in range(len(Face.Tris)):
            edge_length, lengths_to_centre, aspect_ratio = self.ComputeTriLengthMeasurements(Face.Tris, Ys, currentTri,
                                                                                        Face.Centre)
            EdgeLength.append(edge_length)
            LengthsToCentre.append(lengths_to_centre)
            AspectRatio.append(aspect_ratio)
        return EdgeLength, LengthsToCentre, AspectRatio

