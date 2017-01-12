from beaglebone_pins import PWM, GPIO


class Motor(object):
    def __init__(self, pin, directory,gpio_1,gpio_2):
        self.period = 100000
        self.pwm = PWM(pin, directory, self.period)
        self.gpio_1 = GPIO(gpio_1)
        self.gpio_2 = GPIO(gpio_2)

    def setValue(self, value):
        self.gpio_1.set(value > 0)
        self.gpio_2.set(value < 0)
        duty_cycle = int(abs(value) * self.period)
        self.pwm.setDutyCycle(duty_cycle)

    def terminate(self):
        self.gpio_1.set(0)
        self.gpio_2.set(0)
        self.pwm.setDutyCycle(0)
        print "terminated"

if __name__ == "__main__":
    from time import sleep
    m = Motor("P9.14", 'pwm0')
    try:
        while True:
            m.setValue(.1)
            sleep(10)
            m.setValue(.2)
            sleep(10)
    finally:
        m.terminate()
