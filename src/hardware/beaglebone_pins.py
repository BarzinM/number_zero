import subprocess
import re
import os


def getBashOutput(command):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    return process.communicate()[0]  # .decode("utf-8").rstrip()


def getChip(pin):
    import json
    pin = re.sub("\.", "_", pin)
    directory = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(directory, 'pin_index.json')) as data_file:
        data = json.load(data_file)['pinIndex']
    for item in data:
        if item["key"] == pin:
            info = item
            break

    return info["pwm"]["chip"]


def getChipFolder(chip):
    folders = getBashOutput("ls -l /sys/class/pwm").split("\n")
    for folder in folders:
        if chip in folder:
            return re.findall('pwmchip.', folder)[0]


class PWM(object):
    def __init__(self, pin_name, directory,period):
        # config-pin $pin_name pwm
        # print(getBashOutput("config-pin %s pwm" % str(pin_name)))
        chip = getChip(re.sub("\.", "_", pin_name))

        # find corresponding folder in /sys/class/pwm
        dir = getChipFolder(chip)
        dir = os.path.join('/sys/class/pwm', dir)

        dir = os.path.join(dir, directory)
        self.period_dir = os.path.join(dir, 'period')
        self.duty_cycle_dir = os.path.join(dir, 'duty_cycle')
        self.enable_dir = os.path.join(dir, 'enable')

        self.setPeriod(period) # take to setup file
        self.setDutyCycle(0)
        self.enable()

    def enable(self):
        with open(self.enable_dir, "w") as file:
            file.write("1")

    def disable(self):
        self.setDutyCycle(0)
        with open(self.enable_dir, "w") as file:
            file.write("0")

    def setPeriod(self, period):
        with open(self.period_dir, "w") as file:
            file.write(str(period))

    def setDutyCycle(self, duty_cycle):
        with open(self.duty_cycle_dir, "w") as file:
            file.write(str(duty_cycle))

class GPIO(object):
    def __init__(self,gpio_number):
        self.value_dir = '/sys/class/gpio/gpio%i/value'%gpio_number

    def set(self,value):
        with open(self.value_dir,"w") as file:
            file.write("%i"%value)

if __name__ == "__main__":
    from time import sleep

    pin = GPIO(60)
    pin.set(1)
    sleep(1)
    pin.set(0)
    sleep(1)
    pin.set(1)
    sleep(1)
    pin.set(0)
    sleep(1)

    period = 100000
    p14 = PWM("P9.14",'pwm0',period)
    p14.setDutyCycle(period//10)
    sleep(10)
    p14.setDutyCycle(9*period//10)
    sleep(10)
    p14.disable()