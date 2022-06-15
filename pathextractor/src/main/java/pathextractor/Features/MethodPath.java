package pathextractor.Features;

import java.util.LinkedList;

import pathextractor.Common.Common;

/*
 * One path, consist of nodes and edges.
 */
public class MethodPath implements Cloneable {
    public LinkedList<ProgramRelation> path = new LinkedList<>();

    public MethodPath() {

    }

    public MethodPath(ProgramRelation relation) {
        this.path.addLast(relation);
    }

    @SuppressWarnings("unchecked")
    @Override
    public Object clone() {
        MethodPath newPath = null;
        try {
            newPath = (MethodPath) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        newPath.path = (LinkedList<ProgramRelation>) path.clone();
        return newPath;
    }

    public LinkedList<ProgramRelation> getPath() {
        return this.path;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        for (ProgramRelation relation : path) {
            stringBuilder.append(relation.toString());
            stringBuilder.append(",\n");
        }
        return stringBuilder.toString();
    }

    /**
     * Return the extended path.
     */
    public static MethodPath getExtendedPath(ProgramRelation relation, MethodPath Path) {
        MethodPath newPath = (MethodPath) Path.clone();
        ProgramRelation lastRelation = newPath.path.getLast();
        if (!lastRelation.getTarget().equals(relation.getSource())) {
            return null;
        }
        if (lastRelation.type.equals("ASSERT_MC")) {
            if (relation.lineNumber != lastRelation.lineNumber)
                return null;
        }
        if (!relation.getEdge().equals(Common.EmptyEdge) && PathContainsEdge(newPath, relation.getEdge())) {
            return null;
        }
        if (relation.type.equals("VARIABLE")) {
            if (relation.lineNumber != newPath.path.getFirst().lineNumber)
                return null;
        }
        newPath.path.addLast(relation);
        return newPath;
    }

    /**
     * Judge if a methodPath contains given edge.
     */
    public static boolean PathContainsEdge(MethodPath methodPath, Edge edge) {
        for (ProgramRelation programRelation : methodPath.path) {
            if (programRelation.getEdge().equals(edge)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((path == null) ? 0 : path.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        MethodPath other = (MethodPath) obj;
        if (path == null) {
            if (other.path != null) {
                return false;
            }
        } else if (!path.equals(other.path)) {
            return false;
        }
        return true;
    }
}
