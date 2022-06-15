package pathextractor.Features;

import com.fasterxml.jackson.annotation.JsonIgnore;

public class ProgramRelation {
    private Element source;
    private Element target;
    private Edge edge;
    public String type;
    public int lineNumber;

    public ProgramRelation(Element source, Edge edge, Element target, String type, int lineNumber) {
        this.source = source;
        this.target = target;
        this.edge = edge;
        this.type = type;
        this.lineNumber = lineNumber;
    }

    public ProgramRelation() {
    }

    public String toString() {
        return String.format("%s-[%s]->%s", source, edge.toString(), target);
    }

    @JsonIgnore
    public void setType(String type) {
        this.type = type;
    }

    @JsonIgnore
    public Element getSource() {
        return source;
    }

    @JsonIgnore
    public Element getTarget() {
        return target;
    }

    @JsonIgnore
    public Edge getEdge() {
        return edge;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((edge == null) ? 0 : edge.hashCode());
        result = prime * result + ((source == null) ? 0 : source.hashCode());
        result = prime * result + ((target == null) ? 0 : target.hashCode());
        result = prime * result + ((type == null) ? 0 : type.hashCode());
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
        ProgramRelation other = (ProgramRelation) obj;
        if (edge == null) {
            if (other.edge != null) {
                return false;
            }
        } else if (!edge.equals(other.edge)) {
            return false;
        }
        if (source == null) {
            if (other.source != null) {
                return false;
            }
        } else if (!source.equals(other.source)) {
            return false;
        }
        if (target == null) {
            if (other.target != null) {
                return false;
            }
        } else if (!target.equals(other.target)) {
            return false;
        }
        if (type == null) {
            if (other.type != null) {
                return false;
            }
        } else if (!type.equals(other.type)) {
            return false;
        }
        return true;
    }

}
