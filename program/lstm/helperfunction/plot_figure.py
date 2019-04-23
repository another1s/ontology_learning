import matplotlib as plt

class data_Visualization:

    def bar_chart(self, name_list, num_list, figurename):
        plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)
        plt.xlabel("topics")
        plt.ylabel("number of samples")
        plt.savefig(figurename)
        plt.show()
        return

    def accuracy(self, x, accuracy):
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("training accuracy")
        plt.plot(x, accuracy, "b")
        plt.savefig("validation_accuracy.png")
        plt.show()


    def line_chart(self, x, accuracy, loss):
        x0 = x
        accuracy = accuracy
        loss = loss
        a = plt.subplot(211)
        plt.plot(x0, accuracy, "b")
        a.set_title("training accuracy")
        a.set_xlabel = ("training_steps")
        a.set_ylabel = ("accuracy")

        b = plt.subplot(212)
        plt.plot(x0, loss, "r")
        b.set_xlabel("epoch")
        b.set_ylabel("loss")
        plt.savefig("lossandaccuracy.png")
        plt.show()
        return
        plt.savefig("topicdistribution.png")
        plt.show()