from pretrain.data_provider import Data_Form


def generate_data(flnms):
    Data_Form(flnms, 30, 6)


if __name__ == '__main__':
    ute_label = {"frame": "frame",
                 "car_id": "car_id",
                 "left": "left",
                 "top": "top",
                 "width": "width",
                 "height": "height"}
    citysim_label = {"frame": "frameNum",
                     "car_id": "carId",
                     "left": "boundingBox3X",
                     "top": "boundingBox3Y",
                     "right": "boundingBox1X",
                     "bottom": "boundingBox1Y"}
    flnms = {"test1": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-01.csv", "scale": 0.138, "labels": citysim_label},
             "test2": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-02.csv", "scale": 0.138, "labels": citysim_label},
             "test3": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-03.csv", "scale": 0.138, "labels": citysim_label},
             "test4": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-04.csv", "scale": 0.138, "labels": citysim_label},
             "test5": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-05.csv", "scale": 0.138, "labels": citysim_label},
             "test6": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-06.csv", "scale": 0.138, "labels": citysim_label},
             "test7": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-07.csv", "scale": 0.138, "labels": citysim_label},
             "test8": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-08.csv", "scale": 0.138, "labels": citysim_label},
             "test9": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-09.csv", "scale": 0.138, "labels": citysim_label},
             "test10": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-10.csv", "scale": 0.138, "labels": citysim_label},
             "test11": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-11.csv", "scale": 0.138, "labels": citysim_label},
             "test12": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-12.csv", "scale": 0.138, "labels": citysim_label},
             "test13": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-01.csv", "scale": 0.056, "labels": citysim_label},
             "test14": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-02.csv", "scale": 0.056, "labels": citysim_label},
             "test15": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-03.csv", "scale": 0.056, "labels": citysim_label},
             "test25": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-04.csv", "scale": 0.056, "labels": citysim_label},
             "test16": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-06.csv", "scale": 0.056, "labels": citysim_label},
             "test17": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-07.csv", "scale": 0.056, "labels": citysim_label},
             "test18": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-08.csv", "scale": 0.056, "labels": citysim_label},
             "test19": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-09.csv", "scale": 0.056, "labels": citysim_label},
             "test20": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-21.csv", "scale": 0.056, "labels": citysim_label},
             "test21": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-22.csv", "scale": 0.056, "labels": citysim_label},
             "test22": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-23.csv", "scale": 0.056, "labels": citysim_label},
             "test23": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-24.csv", "scale": 0.056, "labels": citysim_label},
             "test24": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-25.csv", "scale": 0.056, "labels": citysim_label},
             "test26": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-12.csv", "scale": 0.056, "labels": citysim_label},
             "test27": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-13.csv", "scale": 0.056, "labels": citysim_label},
             "test28": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-14.csv", "scale": 0.056, "labels": citysim_label},
             "test29": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-17.csv", "scale": 0.056, "labels": citysim_label},
             "test30": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-19.csv", "scale": 0.056, "labels": citysim_label},
             "test31": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-26.csv", "scale": 0.056, "labels": citysim_label},
             "test32": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-27.csv", "scale": 0.056, "labels": citysim_label},
             "test33": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-28.csv", "scale": 0.056, "labels": citysim_label},
             "test34": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-29.csv", "scale": 0.056, "labels": citysim_label},
             }

    flnms_val = {
             "val1": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-10.csv", "scale": 0.056, "labels": citysim_label},
             "val2": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-11.csv", "scale": 0.056, "labels": citysim_label},
             "val3": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-15.csv", "scale": 0.056, "labels": citysim_label},
             "val4": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-16.csv", "scale": 0.056, "labels": citysim_label},
             "val5": {"path": "I://CitySim//ExpresswayA//Expressway A (Weaving Segment)//Trajectories//ExpresswayA-18.csv", "scale": 0.056, "labels": citysim_label},
             }
    freeway_val = {
        "test1": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-01.csv", "scale": 0.138, "labels": citysim_label},
        "test2": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-02.csv", "scale": 0.138, "labels": citysim_label},
        "test3": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-03.csv", "scale": 0.138, "labels": citysim_label},
        "test4": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-04.csv", "scale": 0.138, "labels": citysim_label},
        "test5": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-05.csv", "scale": 0.138, "labels": citysim_label},
        "test6": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-06.csv", "scale": 0.138, "labels": citysim_label},
        "test7": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-07.csv", "scale": 0.138, "labels": citysim_label},
        "test8": {"path": "I://CitySim//FreewayC//Trajectories//FreewayC-08.csv", "scale": 0.138, "labels": citysim_label},
    }
    generate_data(flnms)
    # generate_data(freeway_val)
