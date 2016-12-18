from beaglebone_pins import PWM, GPIO

class Motors(object):
    def __init__(self):
        self.period = 1000000
        self.pwm_1 = PWM("P9.14",'pwm0',self.period)
        self.gpio_1 = GPIO(48)
        self.pwm_2 = PWM("P9.16",'pwm1',self.period)
        self.gpio_2 = GPIO(60)

    def setValues(self,value_1,value_2):
        self.gpio_1.set(value_1>=0)
        self.pwm_1.setDutyCycle(int((1-abs(value_1))*self.period))
        self.gpio_2.set(value_2>=0)
        self.pwm_2.setDutyCycle(int((1-abs(value_2))*self.period))

if __name__ == "__main__":
    from time import sleep
    m = Motors()
    m.setValues(0,0)