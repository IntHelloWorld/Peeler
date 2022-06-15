package pathextractor.Visitors;

import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

/*
 * Visit MethodDeclaration, collect method body string.
 */
public class MethodDeclarationVisitor extends VoidVisitorAdapter<Object> {
    private String methodName;
    private MethodDeclaration mcNode;

    public MethodDeclarationVisitor(String methodName) {
        this.methodName = methodName;
    }

    @Override
    public void visit(MethodDeclaration node, Object arg) {
        visitMethod(node, arg);
        super.visit(node, arg);
    }

    private void visitMethod(MethodDeclaration node, Object obj) {
        if (node.getName().equals(this.methodName)) {
            this.mcNode = node;
        }
    }

    public MethodDeclaration getMethodDeclarationNode() {
        return mcNode;
    }
}
