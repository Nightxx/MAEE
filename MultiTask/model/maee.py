import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
MIN_VAL = -1e10
MAX_VAL = 1e10

class Maee(Layer):
    """
    Mixture of Aspect-Explicit-Experts.
    """

    def __init__(self,
                 units,
                 num_fields,
                 field_expert_num_list,
                 field_expert_index_list,
                 field_expert_type_list,
                 field_expert_boundaries,
                 field_expert_names,
                 num_tasks,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 use_attention_bias=True,
                 use_update_bias=True,
                 user_common_expert_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 attention_activation='sigmoid',
                 update_activation='sigmoid',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 attention_bias_initializer='zeros',
                 update_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 attention_bias_regularizer=None,
                 update_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 attention_bias_constraint=None,
                 update_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 attention_kernel_initializer='VarianceScaling',
                 update_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 attention_kernel_regularizer=None,
                 update_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 attention_kernel_constraint=None,
                 update_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):

        # Hidden nodes parameter
        self.units = units
        self.num_fields = num_fields
        self.field_expert_num_list = field_expert_num_list
        self.field_expert_index_list = field_expert_index_list
        self.field_expert_type_list = field_expert_type_list
        self.field_expert_boundaries = field_expert_boundaries
        self.field_expert_names = field_expert_names
        self.num_tasks = num_tasks

        print(self.field_expert_num_list)
        print(self.field_expert_index_list)
        print(self.field_expert_type_list)
        print(self.field_expert_boundaries)
        print(self.field_expert_names)

        # Weight parameter
        self.common_expert_kernels = None
        self.expert_kernels = None
        self.gate_kernels = None
        self.attention_kernels = None
        self.update_kernels = None

        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.attention_kernel_initializer = initializers.get(attention_kernel_initializer)
        self.update_kernel_initializer = initializers.get(update_kernel_initializer)

        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.attention_kernel_regularizer = regularizers.get(attention_kernel_regularizer)
        self.update_kernel_regularizer = regularizers.get(update_kernel_regularizer)

        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)
        self.attention_kernel_constraint = constraints.get(attention_kernel_constraint)
        self.update_kernel_constraint = constraints.get(update_kernel_constraint)

        # Activation parameter
        self.expert_activation = activations.get(expert_activation)
        self.gate_activation = activations.get(gate_activation)
        self.attention_activation = activations.get(attention_activation)
        self.update_activation = activations.get(update_activation)

        # Bias parameter
        self.common_expert_bias = None
        self.expert_bias = None
        self.gate_bias = None
        self.attention_bias = None
        self.update_bias = None

        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.use_attention_bias = use_attention_bias
        self.use_update_bias = use_update_bias
        self.use_common_expert_bias = user_common_expert_bias

        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.attention_bias_initializer = initializers.get(attention_bias_initializer)
        self.update_bias_initializer = initializers.get(update_bias_initializer)

        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.attention_bias_regularizer = regularizers.get(attention_bias_regularizer)
        self.update_bias_regularizer = regularizers.get(update_bias_regularizer)

        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)
        self.attention_bias_constraint = constraints.get(attention_bias_constraint)
        self.update_bias_constraint = constraints.get(update_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Keras parameter
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        super(Maee, self).__init__(**kwargs)

    def build(self, input_shape):

        assert input_shape is not None and len(input_shape) >= 2

        input_dimension = input_shape[-1]

        self.common_expert_kernel = self.add_weight(
            name='common_expert_kernel',
            shape=(input_dimension, self.units),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        )

        self.common_expert_bias = self.add_weight(
            name='common_expert_bias',
            shape=(self.units,),
            initializer=self.expert_bias_initializer,
            regularizer=self.expert_bias_regularizer,
            constraint=self.expert_bias_constraint,
        )

        # Initialize expert weights (number of input features * number of units per expert)
        self.expert_kernels = [[self.add_weight(
            name='expert_kernel_{}_{}'.format(i, j),
            shape=(input_dimension, self.units),
            initializer=self.expert_kernel_initializer,
            regularizer=self.expert_kernel_regularizer,
            constraint=self.expert_kernel_constraint,
        ) for j in range(self.field_expert_num_list[i])]
            for i in range(self.num_fields)]

        # Initialize expert bias (number of units per expert)
        if self.use_expert_bias:
            self.expert_bias = [[self.add_weight(
                name='expert_bias_{}_{}'.format(i, j),
                shape=(self.units,),
                initializer=self.expert_bias_initializer,
                regularizer=self.expert_bias_regularizer,
                constraint=self.expert_bias_constraint,
            ) for j in range(self.field_expert_num_list[i])]
                for i in range(self.num_fields)]

        # Initialize gate weights (num of tasks * number of input features * number of fields)
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_fields + 1),
            initializer=self.gate_kernel_initializer,
            regularizer=self.gate_kernel_regularizer,
            constraint=self.gate_kernel_constraint
        ) for i in range(self.num_tasks)]

        # Initialize gate bias (number of experts * number of tasks)
        if self.use_gate_bias:
            self.gate_bias = [self.add_weight(
                name='gate_bias_task_{}'.format(i),
                shape=(self.num_fields + 1,),
                initializer=self.gate_bias_initializer,
                regularizer=self.gate_bias_regularizer,
                constraint=self.gate_bias_constraint
            )for i in range(self.num_tasks)]

        # Initialize attention kernel
        self.attention_kernels = [self.add_weight(
            name='attention_kernel_{}'.format(i),
            shape=(2 * self.units, 1),
            initializer=self.attention_kernel_initializer,
            regularizer=self.attention_kernel_regularizer,
            constraint=self.attention_kernel_constraint,
        ) for i in range(self.num_fields)]

        if self.use_attention_bias:
            self.attention_bias = [self.add_weight(
                name='attention_bias_{}'.format(i),
                shape=(1,),
                initializer=self.attention_bias_initializer,
                regularizer=self.attention_bias_regularizer,
                constraint=self.attention_bias_constraint,
            ) for i in range(self.num_fields)]

        # Initialize update kernel
        self.update_kernels = [self.add_weight(
            name='update_kernel_{}'.format(i),
            shape=(2 * self.units, 1),
            initializer=self.update_kernel_initializer,
            regularizer=self.update_kernel_regularizer,
            constraint=self.update_kernel_constraint,
        ) for i in range(self.num_fields)]

        if self.use_update_bias:
            self.update_bias = [self.add_weight(
                name='update_bias_{}'.format(i),
                shape=(1,),
                initializer=self.update_bias_initializer,
                regularizer=self.update_bias_regularizer,
                constraint=self.update_bias_constraint,
            ) for i in range(self.num_fields)]

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dimension})

        super(Maee, self).build(input_shape)

    def call(self, inputs, **kwargs):

        expert_outputs = []
        for i in range(self.num_fields):
            field_experts = []
            mask_list = []
            mask_reverse_list = []
            for j in range(self.field_expert_num_list[i]):

                expert_output = tf.matmul(inputs, self.expert_kernels[i][j])

                if self.use_expert_bias:
                    expert_output = K.bias_add(x=expert_output, bias=self.expert_bias[i][j])
                expert_output = self.expert_activation(expert_output)

                field_experts.append(expert_output)
                if self.field_expert_type_list[i] == "discrete":
                    mask_index = self.field_expert_index_list[i][j]

                    mask_value = tf.expand_dims(tf.greater(inputs[:,mask_index], 0.0),axis=1)

                    mask_reverse_value = tf.logical_not(mask_value)
                    mask_list.append(mask_value)
                    mask_reverse_list.append(mask_reverse_value)
                elif self.field_expert_type_list[i] == "continuous":
                    mask_index = self.field_expert_index_list[i][0]
                    if j == 0:
                        lower = MIN_VAL
                        upper = self.field_expert_boundaries[self.field_expert_names[i]][j]
                    elif j == self.field_expert_num_list[i] - 1:
                        lower = self.field_expert_boundaries[self.field_expert_names[i]][j-1]
                        upper = MAX_VAL
                    else:
                        lower = self.field_expert_boundaries[self.field_expert_names[i]][j-1]
                        upper = self.field_expert_boundaries[self.field_expert_names[i]][j]
                    mask_value = tf.expand_dims(tf.logical_and(
                                                    tf.greater(inputs[:,mask_index], lower),
                                                    tf.logical_not(tf.greater(inputs[:,mask_index], upper))),
                                                axis=1)
                    mask_reverse_value = tf.logical_not(mask_value)
                    mask_list.append(mask_value)
                    mask_reverse_list.append(mask_reverse_value)
            field_experts_stack = tf.stack(field_experts, axis=1)
            mask_list_concat = tf.concat(mask_list, axis=1)

            masked_field_expert = tf.boolean_mask(field_experts_stack, mask_list_concat)
            attention_outputs = []
            for j, field_expert in enumerate(field_experts):
                concat_qk = tf.concat([masked_field_expert, field_expert], axis=1)
                attention_output = tf.matmul(concat_qk, self.attention_kernels[i])
                if self.use_attention_bias:
                    attention_output = K.bias_add(x=attention_output, bias=self.attention_bias[i])
                attention_output = self.attention_activation(attention_output)
                attention_outputs.append(attention_output)
            attention_outputs_stack = tf.stack(attention_outputs, axis=1)

            mask_reverse_list_concat = tf.concat(mask_reverse_list, axis=1)
            reverse_masked_attention_weights = tf.reshape(
                tf.boolean_mask(attention_outputs_stack, mask_reverse_list_concat),
                [-1, self.field_expert_num_list[i] - 1, attention_outputs_stack.get_shape().as_list()[-1]])

            reverse_masked_attention_weights = tf.nn.softmax(reverse_masked_attention_weights, axis=1)


            reverse_masked_attention_weights_tile = tf.tile(reverse_masked_attention_weights,
                                                            [1, 1, masked_field_expert.get_shape().as_list()[-1]])

            reverse_masked_field_experts = tf.reshape(
                tf.boolean_mask(field_experts_stack, mask_reverse_list_concat),
                [-1, self.field_expert_num_list[i] - 1, field_experts_stack.get_shape().as_list()[-1]])

            weighted_reverse_masked_field_experts = tf.multiply(reverse_masked_field_experts,
                                                                reverse_masked_attention_weights_tile)
            aggregated_update_expert = tf.reduce_sum(weighted_reverse_masked_field_experts, axis=1)



            update_input = tf.concat([masked_field_expert, aggregated_update_expert], axis = 1)
            update_output = tf.matmul(update_input, self.update_kernels[i])
            if self.use_update_bias:
                update_output = K.bias_add(x=update_output, bias=self.update_bias[i])
            update_output = self.update_activation(update_output)

            update_weight_tile = tf.tile(update_output, [1, masked_field_expert.get_shape().as_list()[-1]])
            final_field_expert = tf.add(tf.multiply(aggregated_update_expert, update_weight_tile),
                                        tf.multiply(masked_field_expert, 1-update_weight_tile))
            expert_outputs.append(final_field_expert)

        common_expert_output = tf.matmul(inputs, self.common_expert_kernel)
        if self.use_common_expert_bias:
            common_expert_output = K.bias_add(x=common_expert_output, bias=self.common_expert_bias)
        common_expert_output = self.expert_activation(common_expert_output)

        expert_outputs.append(common_expert_output)
        gate_outputs = []

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = K.dot(x=inputs, y=gate_kernel)
            # Add the bias term to the gate weights if necessary
            if self.use_gate_bias:
                gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        final_outputs = []
        for gate_output in gate_outputs:
            gate_output_split = tf.split(gate_output, self.num_fields+1, axis=1)
            weighted_experts = []
            for i, expert_output in enumerate(expert_outputs):
                weighted_experts.append(expert_output * K.repeat_elements(gate_output_split[i], self.units, axis=1))
            final_outputs.append(tf.concat(weighted_experts,axis=1))
            #final_outputs.append(K.sum(weighted_expert_output, axis=2))

        return final_outputs

    def compute_output_shape(self, input_shape):

        assert input_shape is not None and len(input_shape) >= 2

        output_shape = list(input_shape)
        output_shape[-1] = self.units
        output_shape = tuple(output_shape)

        return [output_shape for _ in range(self.num_tasks)]

