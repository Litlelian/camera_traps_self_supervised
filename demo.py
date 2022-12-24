from PyQt5 import QtWidgets

from controller import Dialog_controller

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Dialog_controller()
    window.show()
    sys.exit(app.exec_())