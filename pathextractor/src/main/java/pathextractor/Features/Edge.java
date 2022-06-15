package pathextractor.Features;

import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.StringUtils;

import com.github.javaparser.ast.expr.Expression;
import pathextractor.Common.Common;

public class Edge extends Element {
    private String callerName;
    private List<Expression> args;

    public Edge(String name, String callerName, List<Expression> args) {
        super(name);
        this.callerName = callerName;
        this.args = args;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + ((args == null) ? 0 : args.hashCode());
        result = prime * result + ((callerName == null) ? 0 : callerName.hashCode());
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!super.equals(obj)) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        Edge other = (Edge) obj;
        if (args == null) {
            if (other.args != null) {
                return false;
            }
        } else if (!args.toString().equals(other.args.toString())) {
            return false;
        }
        if (callerName == null) {
            if (other.callerName != null) {
                return false;
            }
        } else if (!callerName.equals(other.callerName)) {
            return false;
        }
        return true;
    }

    public String toString() {
        if (this.equals(Common.EmptyEdge)) {
            return "null";
        }
        List<String> argStrings = args.stream().map(i -> i.toString()).collect(Collectors.toList());
        if (callerName.equals(Common.EmptyString))
            return name + "(" + StringUtils.join(argStrings, ',') + ")";
        else
            return callerName + "." + name + "(" + StringUtils.join(argStrings, ',') + ")";
    }

    public boolean isEmpty() {
        if (this.args.isEmpty() && this.name.equals(Common.EmptyString))
            return true;
        return false;
    }

    public String getName() {
        return name;
    }

    public List<Expression> getArgs() {
        return args;
    }

    public void setArgs(List<Expression> args) {
        this.args = args;
    }

    public String getCallerName() {
        return callerName;
    }

}