class ScoreManager:
    """
     Tool to compute the final score
    """

    def __init__(self,
                 number_drones: int,
                 time_step_limit: int,
                 real_time_limit: int,
                 total_number_wounded_persons: int):

        # 'number_drones' is the number of drones that will be generated in the map
        self.number_drones = number_drones

        # 'time_step_limit' is the number of time steps after which the session will end.
        self.time_step_limit = time_step_limit

        # 'real_time_limit' is the elapsed time (in seconds) after which the session will end.
        self.real_time_limit = real_time_limit

        # 'number_wounded_persons' is the number of wounded persons that should be retrieved by the drones.
        self.total_number_wounded_persons = total_number_wounded_persons

        # weight for the different parts of the score. The sum must be equal to 1.
        self.w_rescue = 0.7
        self.w_exploration = 0.2
        self.w_time = 0.1

    def compute_score(self, number_rescued_persons, score_exploration, rescued_all_time_step):
        """
        Compute the final score out of 100.
        Needed information :
            'number_rescued_persons': number of rescued persons by the drones
            'score_exploration': score of exploration computed by the ExploredMap class
            'rescued_all_time_step': number of time step used by the time all the wounded person are saved
        """
        if self.total_number_wounded_persons > 0:
            percentage_rescue = number_rescued_persons / self.total_number_wounded_persons * 100.0
        else:
            percentage_rescue = 100.0

        if number_rescued_persons == self.total_number_wounded_persons and score_exploration > 97:
            rescued_all_time_step = min(rescued_all_time_step, self.time_step_limit)
            score_time_step = (self.time_step_limit - rescued_all_time_step) / self.time_step_limit * 100.0
        else:
            score_time_step = 0

        score = self.w_rescue * percentage_rescue + \
                self.w_exploration * score_exploration + \
                self.w_time * score_time_step

        return score, percentage_rescue, score_time_step
