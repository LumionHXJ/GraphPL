import torch
from torch import distributions

class ComposeDistribution(distributions.Distribution):
    def __init__(self, *distributions):
        self.distributions = distributions
        super(ComposeDistribution, self).__init__(
            batch_shape=self._calculate_batch_shape(),
            event_shape=self._calculate_event_shape()
        )

    def _calculate_batch_shape(self):
        batch_shapes = [d.batch_shape for d in self.distributions]
        if not all(bs[0] == batch_shapes[0][0] for bs in batch_shapes):
            raise ValueError("batch_shape must same！")
        return batch_shapes[0]

    def _calculate_event_shape(self):
        event_shapes = []
        for d in self.distributions:
            if len(d.event_shape) == 1:
                event_shapes.append(d.event_shape)
            else:
                event_shapes.append(d.batch_shape[1:])
        return torch.Size(sum(event_shapes, ()))

    def sample(self, sample_shape=torch.Size()):
        samples = [d.sample(sample_shape=sample_shape) for d in self.distributions]
        return torch.cat(samples, dim=-1)

    def log_prob(self, value):
        event_sizes = self._calculate_event_shape()
        start = 0
        log_probs = []
        for i, d in enumerate(self.distributions):
            end = start + event_sizes[i]
            log_prob = d.log_prob(value[..., start:end])
            if log_prob.dim() == 1:
                log_prob = log_prob[:, None]
            log_probs.append(log_prob)
            start = end
        total_log_prob = torch.cat(log_probs, dim=-1) 
        return total_log_prob

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ComposeDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.distributions = []
        for d in self.distributions:
            expanded_d = d.expand(batch_shape)
            new.distributions.append(expanded_d)
        super(ComposeDistribution, new).__init__(
            batch_shape=batch_shape,
            event_shape=self.event_shape,
            validate_args=False
        )
        new._validate_args = self._validate_args
        return new
    
    @property
    def mean(self):
        means = []
        for d in self.distributions:
            means.append(d.mean)
        return torch.cat(means, dim=1)