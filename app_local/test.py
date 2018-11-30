def show_movement(movement):

    if movement == 1:  # take_off
        print('                                                  ')
        print('                        XX                        ')
        print('                      XXXXXX                      ')
        print('                    XXXXXXXXXX                    ')
        print('                  XXXXXXXXXXXXXX                  ')
        print('                XXXXXXXXXXXXXXXXXX                ')
        print('              XXXXXXXXXXXXXXXXXXXXXX              ')
        print('            XXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX          ')
        print('                  XXXXXXXXXXXXXX                  ')
        print('                  XXXXXXXXXXXXXX                  ')
        print('                  XXXXXXXXXXXXXX                  ')
        print('                  XXXXXXXXXXXXXX                  ')
        print('                  XXXXXXXXXXXXXX                  ')
        print('                                                  ')
        print('--------------------------------------------------')
    if movement == 2:  # move
        print('                                                  ')
        print('           X             X                        ')
        print('           XXX           XXX                      ')
        print('           XXXXX         XXXXX                    ')
        print('           XXXXXXX       XXXXXXX                  ')
        print('           XXXXXXXXX     XXXXXXXXX                ')
        print('           XXXXXXXXXXX   XXXXXXXXXXX              ')
        print('           XXXXXXXXXXXXX XXXXXXXXXXXXX            ')
        print('           XXXXXXXXXXX   XXXXXXXXXXX              ')
        print('           XXXXXXXXX     XXXXXXXXX                ')
        print('           XXXXXXX       XXXXXXX                  ')
        print('           XXXXX         XXXXX                    ')
        print('           XXX           XXX                      ')
        print('           X             X                        ')
        print('                                                  ')
        print('--------------------------------------------------')
    if movement == 3:  # flip
        print('                                                  ')
        print('                 XXXXX     XXXXX                  ')
        print('              XXXXXX         XXXXXX               ')
        print('           XXXXX                 XXXXX            ')
        print('         XXXXX                     XXXXX          ')
        print('        XXXXX                       XXXXX         ')
        print('        XXXX                         XXXX         ')
        print('       XXXX                           XXXX        ')
        print('        XXXX                         XXXX         ')
        print('        XXXXX                       XXXXX         ')
        print('         XXXXX                     XXXXX          ')
        print('           XXXXX                 XXXXX            ')
        print('              XXXXXX         XXXXXX               ')
        print('                 XXXXX     XXXXX                  ')
        print('                                                  ')
        print('--------------------------------------------------')
    if movement == 4:  # left
        print('                                                  ')
        print('                      XXXX                        ')
        print('                    XXXX                          ')
        print('                  XXXX                            ')
        print('                XXXX                              ')
        print('              XXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('            XXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('            XXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('              XXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('                XXXX                              ')
        print('                  XXXX                            ')
        print('                    XXXX                          ')
        print('                      XXXX                        ')
        print('                                                  ')
        print('--------------------------------------------------')
    if movement == 5:  # right
        print('                                                  ')
        print('                      XXXX                        ')
        print('                        XXXX                      ')
        print('                          XXXX                    ')
        print('                            XXXX                  ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXX                ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXX              ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXX              ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXX                ')
        print('                            XXXX                  ')
        print('                          XXXX                    ')
        print('                        XXXX                      ')
        print('                      XXXX                        ')
        print('                                                  ')
        print('--------------------------------------------------')
    if movement == 6:  # land
        print('                                                  ')
        print('                                                  ')
        print('                   XXXXXXXXXX                     ')
        print('                XXXXXXXXXXXXXXXX                  ')
        print('             XXXXXXXXXXXXXXXXXXXXXX               ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('          XXXXXXXXXXXXXXXXXXXXXXXXXXXX            ')
        print('             XXXXXXXXXXXXXXXXXXXXXX               ')
        print('                XXXXXXXXXXXXXXXX                  ')
        print('                   XXXXXXXXXX                     ')
        print('                                                  ')
        print('                                                  ')
        print('--------------------------------------------------')


for i in range(1, 7):
    show_movement(i)
