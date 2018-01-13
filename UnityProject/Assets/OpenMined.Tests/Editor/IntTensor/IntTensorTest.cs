using System;
using NUnit.Framework;
using OpenMined.Network.Controllers;

namespace OpenMined.Tests.Editor.IntTensorTests
{
    [Category("IntTensorCPUTests")]
    public class IntTensorCPUTest
    {
        private SyftController ctrl;

        [OneTimeSetUp]
        public void Init()
        {
            //Init runs once before running test cases.
            ctrl = new SyftController(null);
        }

        [OneTimeTearDown]
        public void CleanUp()
        {
            //CleanUp runs once after all test cases are finished.
        }

        [SetUp]
        public void SetUp()
        {
            //SetUp runs before all test cases
        }

        [TearDown]
        public void TearDown()
        {
            //SetUp runs after all test cases
        }

        /********************/
        /* Tests Start Here */
        /********************/

        [Test]
        public void Abs()
        {
            int[] shape1 = { 2, 5 };
            int[] data1 = { -1, -2, -3, -4, 5, 6, 7, 8, -999, 10 };
            var tensor1 = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);

            int[] expectedData1 = { 1, 2, 3, 4, 5, 6, 7, 8, 999, 10 };
            int[] shape2 = { 2, 5 };
            var expectedTensor1 = ctrl.intTensorFactory.Create(_data: expectedData1, _shape: shape2);

            var actualTensorAbs1 = tensor1.Abs();

            for (int i = 0; i < actualTensorAbs1.Size; i++)
            {
                Assert.AreEqual(expectedTensor1[i], actualTensorAbs1[i]);
            }
        }

        [Test]
        public void Add()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            var tensorSum = tensor1.Add(tensor2);

            for (int i = 0; i < tensorSum.Size; i++)
            {
                Assert.AreEqual(tensor1[i] + tensor2[i], tensorSum[i]);
            }
        }

        [Test]
        public void Add_()
        {
            float[] data1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            int[] shape1 = { 2, 5 };
            var tensor1 = ctrl.floatTensorFactory.Create(_data: data1, _shape: shape1);

            float[] data2 = { 3, 2, 6, 9, 10, 1, 4, 8, 5, 7 };
            int[] shape2 = { 2, 5 };
            var tensor2 = ctrl.floatTensorFactory.Create(_data: data2, _shape: shape2);

            float[] data3 = { 4, 4, 9, 13, 15, 7, 11, 16, 14, 17 };
            int[] shape3 = { 2, 5 };
            var tensor3 = ctrl.floatTensorFactory.Create(_data: data3, _shape: shape3);

            tensor1.Add(tensor2, inline: true);

            for (int i = 0; i < tensor1.Size; i++)
            {
                Assert.AreEqual(tensor3[i], tensor1[i]);
            }
        }

        [Test]
        public void Trace()
        {
            // test #1
            int[] data1 = {2, 2, 3, 4};
            int[] shape1 = {2, 2};
            var tensor = ctrl.intTensorFactory.Create(_data: data1, _shape: shape1);
            int actual = tensor.Trace();
            int expected = 6;

            Assert.AreEqual(expected, actual);

            // test #2
            int[] data3 = {1, 2, 3};
            int[] shape3 = {3};
            var non2DTensor = ctrl.intTensorFactory.Create(_data: data3, _shape: shape3);
            Assert.That(() => non2DTensor.Trace(),
                Throws.TypeOf<InvalidOperationException>());
        }

        /* closes class and namespace */
    }
}