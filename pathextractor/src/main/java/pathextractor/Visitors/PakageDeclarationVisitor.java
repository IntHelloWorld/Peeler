package pathextractor.Visitors;

import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class PakageDeclarationVisitor extends VoidVisitorAdapter<Object> {
    private PackageDeclaration pakageDeclaration;

    @Override
    public void visit(PackageDeclaration node, Object arg) {
        pakageDeclaration = node;
        super.visit(node, arg);
    }

    public PackageDeclaration getPakageDeclaration() {
        return pakageDeclaration;
    }
}
